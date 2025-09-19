import pickle
import sys
from collections import Counter

def analyze_pickle_structure(file_path, sample_size=5):
    """
    Analyze the structure of a pickle file with special handling for large lists of tuples.
    
    Args:
        file_path: Path to the pickle file
        sample_size: Number of items to sample from large collections
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Structure of pickle file: {file_path}")
        print("=" * 50)
        
        if isinstance(data, list):
            print(f"Main structure: List with {len(data)} items")
            
            if len(data) > 0:
                # Analyze the first few items
                print(f"\nSample of first {min(sample_size, len(data))} items:")
                for i, item in enumerate(data[:sample_size]):
                    if isinstance(item, tuple):
                        print(f"  Item {i}: Tuple with {len(item)} elements")
                        for j, element in enumerate(item):
                            print(f"    Element {j}: Type={type(element).__name__}, Length={len(element) if hasattr(element, '__len__') else 'N/A'}")
                            if isinstance(element, str):
                                print(f"      First 100 chars: {element[:100]}")
                    else:
                        print(f"  Item {i}: Type={type(item).__name__}")
                
                # Collect statistics about tuple lengths
                if len(data) > sample_size:
                    tuple_lengths = [len(item) for item in data if isinstance(item, tuple)]
                    if tuple_lengths:
                        counter = Counter(tuple_lengths)
                        print(f"\nTuple length distribution:")
                        for length, count in counter.most_common():
                            print(f"  Tuples with {length} elements: {count} items ({count/len(data):.1%})")
            
        else:
            print(f"Main structure: {type(data).__name__}")
            
    except Exception as e:
        print(f"Error loading pickle file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_pickle_structure(sys.argv[1])
    else:
        print("Please provide the path to a pickle file as an argument.")
