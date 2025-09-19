import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# For initialization functions or wrapper stubs (if used)
from scannet_pytorch.preprocessing.pipelines import padd_matrix
#from scannet_pytorch import wrappers
from scannet_pytorch.preprocessing import PDB_processing



def distance_pcs(cloud1, cloud2, squared=False):
    # cloud1: [B, L, 3], cloud2: [B, K, 3]
    diff = cloud1.unsqueeze(-2) - cloud2.unsqueeze(-3)  # → [B, L, K, 3]
    dist_sq = (diff ** 2).sum(dim=-1)  # → [B, L, K]
    return dist_sq if squared else torch.sqrt(dist_sq + 1e-8)


class FrameBuilder(nn.Module):
    def __init__(self, order='1', dipole=False):
        super(FrameBuilder, self).__init__()
        self.order = order
        self.dipole = dipole
        self.epsilon = 1e-6

        # Reference axis for numerical stability
        self.register_buffer('xaxis', torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))
        self.register_buffer('yaxis', torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32))
        self.register_buffer('zaxis', torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32))

    def forward(self, inputs, mask=None):
        points, triplets = inputs  # points: [B, L, 3], triplets: [B, L, 3] or [B, L, 4]

        B, L, _ = triplets.shape
        triplets = torch.clamp(triplets, 0, points.size(-2) - 1)

        # Vector differences
        delta_10 = torch.gather(points, 1, triplets[:, :, 1:2].expand(-1, -1, 1, 3)) - \
                   torch.gather(points, 1, triplets[:, :, 0:1].expand(-1, -1, 1, 3))
        delta_20 = torch.gather(points, 1, triplets[:, :, 2:3].expand(-1, -1, 1, 3)) - \
                   torch.gather(points, 1, triplets[:, :, 0:1].expand(-1, -1, 1, 3))

        if self.order in ['2', '3']:
            delta_10, delta_20 = delta_20, delta_10

        centers = torch.gather(points, 1, triplets[:, :, 0:1].expand(-1, -1, 1, 3)).squeeze(2)

        zaxis = (delta_10 + self.epsilon * self.zaxis).squeeze(2)
        zaxis = zaxis / (zaxis.norm(dim=-1, keepdim=True) + self.epsilon)

        yaxis = torch.cross(zaxis, delta_20.squeeze(2), dim=-1)
        yaxis = (yaxis + self.epsilon * self.yaxis)
        yaxis = yaxis / (yaxis.norm(dim=-1, keepdim=True) + self.epsilon)

        xaxis = torch.cross(yaxis, zaxis, dim=-1)
        xaxis = (xaxis + self.epsilon * self.xaxis)
        xaxis = xaxis / (xaxis.norm(dim=-1, keepdim=True) + self.epsilon)

        if self.order == '3':
            xaxis, yaxis, zaxis = zaxis, xaxis, yaxis

        if self.dipole:
            dipole = torch.gather(points, 1, triplets[:, :, 3:4].expand(-1, -1, 1, 3)) - \
                     torch.gather(points, 1, triplets[:, :, 0:1].expand(-1, -1, 1, 3))
            dipole = (dipole + self.epsilon * self.zaxis).squeeze(2)
            dipole = dipole / (dipole.norm(dim=-1, keepdim=True) + self.epsilon)
            frames = torch.stack([centers, xaxis, yaxis, zaxis, dipole], dim=-2)
        else:
            frames = torch.stack([centers, xaxis, yaxis, zaxis], dim=-2)

        # Optional masking
        if mask is not None and isinstance(mask, (list, tuple)) and mask[-1] is not None:
            m = mask[-1].float().unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
            frames = frames * m

        return frames

def distance(coordinates1, coordinates2, squared=False, ndims=3):
    # coordinates1: [..., D], coordinates2: [..., D]
    diff = coordinates1[..., :ndims].unsqueeze(-2) - coordinates2[..., :ndims].unsqueeze(-3)  # [..., L, K, ndims]
    dist_sq = (diff ** 2).sum(dim=-1)
    return dist_sq if squared else torch.sqrt(dist_sq + 1e-8)


def euclidian_to_spherical(x, return_r=True, cut='2pi', eps=1e-8):
    r = torch.sqrt(torch.sum(x ** 2, dim=-1))
    theta = torch.acos(x[..., 2] / (r + eps))  # z / r
    phi = torch.atan2(x[..., 1], x[..., 0] + eps)  # y / x

    if cut == '2pi':
        phi = phi + (phi < 0).float() * (2 * np.pi)

    if return_r:
        return torch.stack([r, theta, phi], dim=-1)
    else:
        return torch.stack([theta, phi], dim=-1)


class LocalNeighborhood(nn.Module):
    def __init__(self, Kmax=10, coordinates=['euclidian'], self_neighborhood=True,
                 index_distance_max=None, nrotations=1):
        super(LocalNeighborhood, self).__init__()

        self.Kmax = Kmax
        self.coordinates = coordinates
        self.self_neighborhood = self_neighborhood
        self.index_distance_max = index_distance_max
        self.nrotations = nrotations
        self.epsilon = 1e-10
        self.big_distance = 1000.0

        # Validate coordinate types
        for coord in coordinates:
            assert coord in ['distance', 'index_distance', 'euclidian', 'ZdotZ', 'ZdotDelta', 'dipole_spherical']

        # Set format expectations
        self.first_format = []
        self.second_format = []

        if any(coord in coordinates for coord in ['euclidian', 'ZdotZ', 'ZdotDelta']):
            self.first_format.append('frame')
            self.second_format.append('frame' if self.self_neighborhood or any(c in coordinates for c in ['ZdotZ', 'ZdotDelta']) else 'point')
        elif 'distance' in coordinates:
            self.first_format.append('point')
            self.second_format.append('point')
        if 'index_distance' in coordinates:
            self.first_format.append('index')
            self.second_format.append('index')

        # Compute coordinate embedding dimension
        dim = 0
        for coord in coordinates:
            if coord == 'euclidian':
                dim += 3
            elif coord == 'dipole_spherical':
                dim += 2
            elif coord == 'ZdotDelta':
                dim += 2
            else:
                dim += 1
        self.coordinates_dimension = dim

        # Rotation setup
        if self.nrotations > 1:
            assert self.coordinates == ['euclidian'], 'Rotations only supported for euclidian coordinates'
            phis = np.arange(nrotations) / nrotations * 2 * np.pi
            rotations = np.zeros((nrotations, 3, 3), dtype=np.float32)
            rotations[:, 0, 0] = np.cos(phis)
            rotations[:, 1, 1] = np.cos(phis)
            rotations[:, 0, 1] = -np.sin(phis)
            rotations[:, 1, 0] = np.sin(phis)
            rotations[:, 2, 2] = 1
            self.register_buffer("rotations", torch.tensor(rotations))

        #print("embed_atom.first_format:", self.first_format)
        #print("embed_atom.second_format:", self.second_format)

    def forward(self, inputs, mask=None):
        # Unpack inputs according to declared formats
        if mask is None:
            mask = [None for _ in inputs]

        def get_input(kind, from_first=True):
            fmt = self.first_format if from_first else self.second_format
            idx = fmt.index(kind)
            if self.self_neighborhood or from_first:
                return inputs[idx]
            else:
                return inputs[len(self.first_format) + idx]

        first_frame = get_input('frame') if 'frame' in self.first_format else None
        second_frame = first_frame if self.self_neighborhood else get_input('frame', from_first=False) if 'frame' in self.second_format else None

        first_point = get_input('point') if 'point' in self.first_format else None
        second_point = first_point if self.self_neighborhood else get_input('point', from_first=False) if 'point' in self.second_format else None

        first_index = get_input('index') if 'index' in self.first_format else None
        second_index = first_index if self.self_neighborhood else get_input('index', from_first=False) if 'index' in self.second_format else None

        second_attributes = inputs[-(len(inputs) - len(self.first_format) - (0 if self.self_neighborhood else len(self.second_format))):]

        first_mask = mask[0]
        if first_mask is not None and first_frame is not None:
            first_mask = first_mask[:, :, 1]
        second_mask = first_mask if self.self_neighborhood else mask[len(self.first_format)]
        if second_mask is not None and second_frame is not None:
            second_mask = second_mask[:, :, 1]
        irrelevant_seconds = (1 - second_mask.float()).unsqueeze(1) if second_mask is not None else None

        # Select neighborhood centers
        if first_frame is not None:
            first_center = first_frame[:, :, 0]
            ndims = 3
        elif first_point is not None:
            first_center = first_point
            ndims = 3
        else:
            first_center = first_index.float()
            ndims = 1

        if second_frame is not None:
            second_center = second_frame[:, :, 0]
        elif second_point is not None:
            second_center = second_point
        else:
            second_center = second_index.float()

        # Pairwise distance
        diff = first_center.unsqueeze(-2) - second_center.unsqueeze(-3)
        distance_sq = (diff[..., :ndims] ** 2).sum(dim=-1)

        if irrelevant_seconds is not None:
            distance_sq += irrelevant_seconds * self.big_distance

        max_neighbors = min(self.Kmax, distance_sq.shape[-1])
        neighbors = torch.topk(distance_sq, k=max_neighbors, dim=-1, largest=False).indices.unsqueeze(-1)


    # Gather attributes of neighbors
        neighbors_attributes = [
            torch.gather(attr.unsqueeze(2).expand(-1, -1, neighbors.shape[2], -1), 1,
                 neighbors.expand(-1, -1, -1, attr.shape[-1]))
            for attr in second_attributes
        ]
        
    # --- Coordinate encoding ---
        neighbor_coordinates = []

        if 'euclidian' in self.coordinates:
            delta = torch.gather(second_center, 1, neighbors.expand(-1, -1, -1, 3)) - first_center.unsqueeze(-2)
            euclidian = torch.einsum('blkc,blcr->blkr', delta, first_frame[:, :, 1:4])  # frame projection
            if self.nrotations > 1:
                euclidian = torch.matmul(euclidian.unsqueeze(-2), self.rotations).squeeze(-2)
                neighbors_attributes = [attr.unsqueeze(-2) for attr in neighbors_attributes]
            neighbor_coordinates.append(euclidian)

        if 'dipole_spherical' in self.coordinates:
            dipole_vec = torch.gather(second_frame[:, :, -1], 1, neighbors.expand(-1, -1, -1, 3))
            dipole_proj = torch.einsum('blkc,blcr->blkr', dipole_vec, first_frame[:, :, 1:4])
            dipole_sph = euclidian_to_spherical(dipole_proj, return_r=False)
            neighbor_coordinates.append(dipole_sph)

        if 'distance' in self.coordinates:
            dist = torch.sqrt(torch.gather(distance_sq.unsqueeze(-1), 2, neighbors) + self.epsilon)
            neighbor_coordinates.append(dist)

        if 'ZdotZ' in self.coordinates:
            z1 = first_frame[:, :, -1]
            z2 = second_frame[:, :, -1]
            z2_neigh = torch.gather(z2, 1, neighbors.expand(-1, -1, -1, 3))
            dot = (z1.unsqueeze(-2) * z2_neigh).sum(dim=-1, keepdim=True)
            neighbor_coordinates.append(dot)

        if 'ZdotDelta' in self.coordinates:
            z1 = first_frame[:, :, -1]
            z2 = second_frame[:, :, -1]
            delta = (torch.gather(second_center, 1, neighbors.expand(-1, -1, -1, 3)) - first_center.unsqueeze(-2))
            dist = torch.sqrt(torch.gather(distance_sq, 2, neighbors.squeeze(-1)).unsqueeze(-1) + self.epsilon)
            delta = delta / (dist + self.epsilon)
            zdot = (z1.unsqueeze(-2) * delta).sum(dim=-1, keepdim=True)
            dot2 = (delta * torch.gather(z2, 1, neighbors.expand(-1, -1, -1, 3))).sum(dim=-1, keepdim=True)
            neighbor_coordinates.append(dot2)
            neighbor_coordinates.append(zdot)

        if 'index_distance' in self.coordinates:
            diff = torch.abs(first_index.unsqueeze(-1) - torch.gather(second_index, 1, neighbors.squeeze(-1)))
            index_dist = diff.float().unsqueeze(-1)
            if self.index_distance_max is not None:
                index_dist = torch.clamp(index_dist, 0, self.index_distance_max)
            neighbor_coordinates.append(index_dist)

        # Combine all coordinates
        neighbor_coordinates = torch.cat(neighbor_coordinates, dim=-1)

        # Apply masks
        if first_mask is not None:
            m = first_mask.float().unsqueeze(-1).unsqueeze(-1)
            if self.nrotations > 1:
                m = m.unsqueeze(-1)
            neighbor_coordinates = neighbor_coordinates * m
            neighbors_attributes = [attr * m for attr in neighbors_attributes]

        return [neighbor_coordinates] + neighbors_attributes


def get_LocalNeighborhood(inputs,neighborhood_params,flat=False,n_samples=100,padded=False,attributes=None):
    B = len(inputs[0])
    if n_samples is not None:
        b = min(n_samples, B)
    else:
        b = B

    if padded:
        Lmaxs = [inputs_.shape[1] for inputs_ in inputs]
        inputs = [inputs_[:b] for inputs_ in inputs]
        if attributes is not None:
            attributes = attributes[:b]
    else:
        Lmaxs = [ max([len(input_) for input_ in inputs_[:b]] ) for inputs_ in  inputs ]
        inputs = [
        np.stack(
        [padd_matrix(input,Lmax=Lmax,padding_value=0) for input in inputs_[:b]],
        axis =0
        )
        for Lmax,inputs_ in zip(Lmaxs,inputs)

        ]
        if attributes is not None:
            attributes = np.stack([padd_matrix(attribute,Lmax=Lmaxs[-1],padding_value=0.) for attribute in attributes[:b]],
                                  axis=0)
    if attributes is not None:
        inputs.append(attributes)
    else:
        inputs.append(  np.ones([b, Lmaxs[0], 1], dtype=np.float32) )

    keras_inputs  = [Input(shape=inputs_.shape[1:]) for inputs_ in inputs]
    masked_keras_inputs = [Masking()(keras_inputs_) for keras_inputs_ in keras_inputs]
    local_coordinates, local_attributes = LocalNeighborhood(
        **neighborhood_params)(masked_keras_inputs)

    first_layer = Model(
        inputs=keras_inputs, outputs=[local_coordinates,local_attributes])

    local_coordinates,local_attributes = first_layer.predict(inputs, batch_size=10)

    if flat:
        d = local_coordinates.shape[-1]
        nattributes = local_attributes.shape[-1]
        local_coordinates = local_coordinates[local_coordinates.max(
            -1).max(-1) > 0].reshape([-1, d])
        local_attributes = local_attributes[local_attributes.max(-1)>0].reshape([-1,nattributes])
    if attributes is not None:
        return local_coordinates,local_attributes
    else:
        return local_coordinates


def get_Frames(inputs,n_samples=None,padded=False,order='1',dipole=False,Lmax=None):
    B = len(inputs[0])
    if n_samples is not None:
        b = min(n_samples, B)
    else:
        b = B

    nindices = inputs[0][0].shape[-1]

    if padded:
        triplets_ = inputs[0][:b]
        clouds_ = inputs[1][:b]
        Lmax = inputs[0].shape[1]
        Lmax2 = inputs[1].shape[1]
    else:
        if Lmax is not None:
            Lmax = min(max([len(input_) for input_ in inputs[0][:b]] ), Lmax)
        else:
            Lmax = max([len(input_) for input_ in inputs[0][:b]] )
        Lmax2 = max([len(input_) for input_ in inputs[1][:b]] )
        Ls = [len(x) for x in inputs[0][:b]]
        triplets_ = np.zeros([b,Lmax, nindices ],dtype=np.int32)
        clouds_ = np.zeros([b,Lmax2,3],dtype=np.float32)

        for b_ in range(b):
            padd_matrix(inputs[0][b_], padded_matrix=triplets_[b_], padding_value=-1)
            padd_matrix(inputs[1][b_], padded_matrix=clouds_[b_], padding_value=0)

    inputs_ = [triplets_,clouds_]

    triplets = Input(shape=[Lmax, nindices], dtype="int32",name='triplets')
    clouds = Input(shape=[Lmax2, 3], dtype="float32",name='clouds')
    masked_triplets = Masking(mask_value=-1, name='masked_triplets')(triplets)
    masked_clouds = Masking(mask_value=0.0, name='masked_clouds')(clouds)
    frames = FrameBuilder(name='frames',order=order,dipole=dipole)([masked_clouds,masked_triplets])
    first_layer = Model(
        inputs=[triplets,clouds], outputs=frames)
    frames_ = first_layer.predict(inputs_)
    if not padded:
        frames_ = wrappers.truncate_list_of_arrays(frames_,Ls)
    return frames_



def initialize_GaussianKernel_for_NeighborhoodEmbedding(
        inputs, N,
        covariance_type='diag',
        neighborhood_params = {'Kmax':10,'coordinates':['euclidian'],'nrotations':1,'index_distance_max':None,'self_neighborhood':True},
        from_triplets = False, Dmax = None, n_samples=None,padded=True,order='1',dipole=False,n_init=10):

    if from_triplets:
        frames = get_Frames(inputs[:2],n_samples=n_samples,padded=padded,order=order,dipole=dipole)
        inputs = [frames] + inputs[2:]

    local_coordinates = get_LocalNeighborhood(inputs, neighborhood_params, flat=True, n_samples=n_samples,padded=padded)

    if Dmax is not None:
        if 'euclidian' in neighborhood_params['coordinates']:
            d = np.sqrt((local_coordinates[:,:3]**2).sum(-1))
        else:
            d = local_coordinates[:,0]
        local_coordinates = local_coordinates[d <= Dmax]
    if 'index_distance' in neighborhood_params['coordinates']:
        reg_covar = 1e0
    else:
        reg_covar= 1e-2
    return initialize_GaussianKernel(local_coordinates, N,covariance_type=covariance_type,reg_covar=reg_covar,n_init=n_init)



def initialize_Embedding_for_NeighborhoodAttention(
        inputs, labels,N=16,covariance_type='full',dense=None,
            nsamples=100,
            epochs=10,
        neighborhood_params={
            'Kmax': 32,
            'coordinates': ['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'],
            'nrotations': 1,
            'index_distance_max': 16,
            'self_neighborhood': True},
        from_triplets=False, n_samples=None, padded=True,Dmax=None,order='1',dipole=False,n_init=10):
    '''
    labels in binary format.
    '''
    if nsamples is not None:
        inputs = [input[:nsamples] for input in inputs]
        labels = labels[:nsamples]

    Ls = [label.shape[0] for label in labels]
    if padded:
        inputs = wrappers.truncate_list_of_arrays(inputs, Ls)
        labels = wrappers.truncate_list_of_arrays(labels, Ls)


    if from_triplets:
        frames = get_Frames(inputs[:2],order=order,dipole=dipole, n_samples=n_samples, padded=False)
        frames = wrappers.truncate_list_of_arrays(frames, Ls)
        inputs = [frames] + inputs[2:]

    mu_labels = np.concatenate(labels,axis=0)[:,-1].mean()
    local_coordinates,local_attributes = get_LocalNeighborhood(inputs,neighborhood_params,flat=False,padded=False,attributes=labels)
    local_coordinates = np.concatenate(wrappers.truncate_list_of_arrays(local_coordinates, Ls), axis=0)
    local_attributes = np.concatenate(wrappers.truncate_list_of_arrays(local_attributes, Ls),axis=0)
    features = local_coordinates.reshape([-1,local_coordinates.shape[-1]] )
    target = ( (local_attributes[:,:1,-1] * local_attributes[:,:,-1]).flatten() - mu_labels**2)/(mu_labels - mu_labels**2)
    if Dmax is not None:
        mask = features[:,0] < Dmax
        features = features[mask]
        target = target[mask]
    initial_values = initialize_GaussianKernel(features, N, covariance_type=covariance_type,n_init=n_init)

    model = Sequential()
    model.add(GaussianKernel(N, covariance_type=covariance_type,
                      initial_values=initial_values, name='graph_embedding_GaussianKernel'))
    if dense is not None:
        model.add(Dense(dense, activation='tanh',
                        name='graph_embedding_dense', use_bias=False))
        model.add(Dense(1, activation=None,use_bias=False, name='graph_embedding_dense_final'))
    else:
        model.add(Dense(1, activation=None, use_bias=False, name='graph_embedding_dense'))

    model.compile(loss='MSE', optimizer='adam')
    model.fit(features, target, epochs=epochs, batch_size=1024)
    model_params = dict([(layer.name, layer.get_weights())
                         for layer in model.layers])
    return model_params






if __name__ == '__main__':
# %%
    import matplotlib
    matplotlib.use('module://backend_interagg')
    import matplotlib.pyplot as plt
    import PDB_utils2
    import Bio.PDB
    import pipelines
    import numpy as np
    import wrappers
    import format_dockground



    with_atom = True
    aa_frames = 'quadruplet'
    order = '3'
    dipole = True

    pipeline = pipelines.ScanNetPipeline(
                                                        with_aa=True,
                                                        with_atom=with_atom,
                                                        aa_features='sequence',
                                                        atom_features='type',
                                                        aa_frames=aa_frames,
    )


    PDB_folder = '/Users/jerometubiana/PDB/'
    pdblist = Bio.PDB.PDBList()
    # list_pdbs = ['11as_A',
    #              '137l_B',
    #              '13gs_A',
    #              '1a05_A',
    #              '1a09_A',
    #              '1a0d_A',
    #              '1a0e_A',
    #              '1a0f_A',
    #              '1a0g_B',
    #              '1a0o_B']

    nmax = 10

    list_origins, list_sequences,list_resids,list_labels = format_dockground.read_labels('/Users/jerometubiana/Downloads/interface_labels_train.txt',nmax=nmax,label_type='int')

    inputs = []
    for origin in list_origins:
        pdb = origin[:4]
        chain = origin.split('_')[-1]
        name = pdblist.retrieve_pdb_file(pdb, pdir=PDB_folder)
        struct, chains = PDBio.load_chains(pdb_id=pdb, chain_ids=[(0, chain)], file=PDB_folder + '%s.cif' % pdb)
        inputs.append(pipeline.process_example(chains))
    inputs = [np.array([input[k] for input in inputs])
              for k in range(len(inputs[0]))]
    outputs = [ np.stack([label <5,label >=5],axis = -1) for label in list_labels]


    frames = get_Frames([inputs[0],inputs[3]],order=order,dipole=dipole) # Valide 24/01/2021

    plt.plot(frames[0][:,0,:]); plt.show() # Check centers.
    plt.plot(frames[0][:, 1, :]); plt.show()  # Check unit vectors.
    for i in range(3): # Check orthonormality.
        for j in range(3):
            print('Dot product',i,j , np.abs( (frames[0][:, 1+i, :] * frames[0][:,1+j,:]).sum(-1)  ).max() )

    local_coordinates = get_LocalNeighborhood([frames],padded=False,neighborhood_params={
        'coordinates': ['euclidian','dipole_spherical'],
        'Kmax':10
    })

    plt.hist( local_coordinates.flatten(),bins=100 ); plt.show() # Check scales.
    plt.matshow(np.sqrt( (local_coordinates[0]**2).sum(-1) ) ,aspect='auto'); plt.colorbar(); plt.show() # Check order and padding.
#%%
    local_coordinates = get_LocalNeighborhood([frames,inputs[2]],padded=False,neighborhood_params={
        'coordinates': ['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'],
        'Kmax':10,
        'index_distance_max': 16,
    })

    plt.matshow(local_coordinates[0,:,:,0],aspect='auto'); plt.colorbar(); plt.show()
    for i in range(local_coordinates.shape[-1]):
        plt.hist(local_coordinates[...,i].flatten(),bins=100); plt.show()

    plt.matshow(local_coordinates[0,:,:,-1],aspect='auto'); plt.colorbar(); plt.show()

#%%

    params = initialize_GaussianKernel_for_NeighborhoodEmbedding(
        [inputs[0],inputs[3]], 16,
        covariance_type='diag',
        neighborhood_params = {
            'Kmax':10,
            'coordinates':['euclidian','dipole_spherical'],'nrotations':1,'index_distance_max':None,'self_neighborhood':True},
        from_triplets = True,
        dipole=dipole,
        padded=False)

    plt.plot(params[0].T); plt.show()
    plt.hist(params[1].flatten(), bins=20); plt.show()

#%%

    params = initialize_GaussianKernel_for_NeighborhoodEmbedding(
        [inputs[0], inputs[3],inputs[2]], 16,
        covariance_type='diag',
        neighborhood_params={
            'Kmax': 10,
            'coordinates': ['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'], 'nrotations': 1, 'index_distance_max':  8, 'self_neighborhood':True},
        from_triplets=True,
        padded=False)

    plt.plot(params[0].T)
    plt.show()
    plt.hist(params[1].flatten(), bins=20)
    plt.show()

#%%

    params = initialize_GaussianKernel_for_NeighborhoodEmbedding(
        [inputs[2]], 16,
        covariance_type='diag',
        neighborhood_params = {
            'Kmax':10,
            'coordinates':['index_distance'],'nrotations':1,'index_distance_max':16,'self_neighborhood':True},
        from_triplets = False,
        padded=False)


#%%
    model_params = initialize_Embedding_for_NeighborhoodAttention([inputs[0], inputs[3],inputs[2]],outputs,padded=False,from_triplets=True)

    from keras.models import Sequential
    import embeddings
    model = Sequential()
    model.add(GaussianKernel(16, covariance_type='full',
                      initial_values=model_params['graph_embedding_GaussianKernel'], name='graph_embedding_GaussianKernel',input_shape=(5,)))
    model.add(Dense(1, activation='linear',use_bias=False, name='graph_embedding_dense'))
    model.layers[-1].set_weights(model_params['graph_embedding_dense'])

    local_coordinates_flat = local_coordinates[local_coordinates.max(-1).max(-1)>0]
    graph_value = model.predict(local_coordinates_flat.reshape([-1,5]))
    plt.scatter(local_coordinates_flat[:,:,0].flatten(),graph_value[:,0],s=1,c=graph_value[:,-1]); plt.show()




    #%% Check consistency with wrapper.
    import wrappers
    import numpy as np


    all_triplets = []
    all_clouds = []

    B = 100
    for b in range(B):
        L = np.random.randint(5,high=21)
        random_direction1 = np.random.randn(3)
        random_direction2 = np.random.randn(3)
        points = np.random.randn(L,3)
        cloud = np.concatenate([
            points,
            points + random_direction1[np.newaxis],
            points + random_direction2[np.newaxis]
        ], axis=0)
        triplet = np.stack([
            np.arange(L),
            np.arange(L)+L,
            np.arange(L)+2*L], axis=-1)
        all_clouds.append(cloud)
        all_triplets.append(triplet)


    def keras_frames(Lmax=20):
        from keras.engine.base_layer import Layer
        from keras import backend as K
        import tensorflow as tf
        import numpy as np
        from keras.layers import Input, Masking, Dense
        from keras.models import Model
        cloud = Input(shape=(3*Lmax,3),dtype="float32")
        triplet = Input(shape=(Lmax, 3), dtype="int32")
        masked_cloud = Masking(mask_value=0.0, name='masked_indices_atom')(cloud)
        masked_triplets = Masking(mask_value=-1, name='masked_triplets_atom')(triplet)
        frames = FrameBuilder()([cloud,triplet])
        model = Model(inputs=[triplet,cloud],outputs=frames)
        return model


    model = wrappers.grouped_Predictor_wrapper(keras_frames,
                                               Lmax=20,
                                               multi_inputs=True,
                                               input_type=['triplets','points'],
                                               Lmaxs=[20,3*20])


    all_frames = model.predict([all_triplets,all_clouds],return_all=True,batch_size=1)

    for i,frame in enumerate(all_frames):
        print(i,(frame[:,1:].max(0)-frame[:,1:].min(0) ).max()  )
