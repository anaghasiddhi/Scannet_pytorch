import argparse
import os
import torch
import esm
from tqdm import tqdm

def load_fasta(fasta_path):
    sequences = []
    with open(fasta_path, "r") as f:
        name, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    sequences.append((name, "".join(seq)))
                name = line[1:]
                seq = []
            else:
                seq.append(line)
        if name is not None:
            sequences.append((name, "".join(seq)))
    return sequences

def main(args):
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    sequences = load_fasta(args.fasta)
    print(f"[INFO] Loaded {len(sequences)} sequences from {args.fasta}")

    batch_size = args.batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size)):
            batch = sequences[i:i + batch_size]
            # Truncate sequences BEFORE tokenization
            if args.truncation_seq_length:
                batch = [
                    (label, seq[:args.truncation_seq_length])
                    for (label, seq) in batch
                ]

            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)
         

            # Additional safety: hard truncate token sequence length to (BOS + 1024 + EOS)
            max_token_len = args.truncation_seq_length + 2  # BOS + EOS
            if batch_tokens.size(1) > max_token_len:
                batch_tokens = batch_tokens[:, :max_token_len]
            # Calculate raw lengths and filter out sequences > max_length
            max_len = args.truncation_seq_length
            lengths = (batch_tokens != alphabet.padding_idx).sum(dim=1) - 2  # exclude BOS/EOS
            if (lengths > max_len).any():
                # Filter batch to only valid length entries
                valid_idx = (lengths <= max_len).nonzero(as_tuple=True)[0]
                batch_tokens = batch_tokens[valid_idx]
                batch_labels = [batch_labels[k] for k in valid_idx.tolist()]
                batch_strs = [batch_strs[k] for k in valid_idx.tolist()]
                if len(batch_tokens) == 0:
                    continue  # skip if batch fully invalid

            results = model(batch_tokens, repr_layers=[args.repr_layers], return_contacts=False)
            representations = results["representations"][args.repr_layers]

            for j, (label, tokens, reps) in enumerate(zip(batch_labels, batch_tokens, representations)):
                seq_len = (tokens != alphabet.padding_idx).sum().item() - 2  # Exclude BOS/EOS
                if args.truncation_seq_length and seq_len > args.truncation_seq_length:
                    seq_len = args.truncation_seq_length
                    reps = reps[1:1 + seq_len]  # Exclude BOS, truncate
                else:
                    reps = reps[1:1 + seq_len]  # Exclude BOS

                out_data = {}
                if args.include_mean:
                    out_data["mean_representations"] = reps.mean(0)
                if args.include_per_tok:
                    out_data["token_representations"] = reps

                torch.save(out_data, os.path.join(args.out, f"{label}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repr_layers", type=int, default=33)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--truncation_seq_length", type=int, default=None)
    parser.add_argument("--include_mean", action="store_true")
    parser.add_argument("--include_per_tok", action="store_true")
    args = parser.parse_args()
    main(args)
