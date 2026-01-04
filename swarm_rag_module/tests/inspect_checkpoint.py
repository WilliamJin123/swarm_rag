import sys
import os
import pickle

# Ensure local modules are visible
sys.path.append(os.getcwd())

def inspect(path):
    print(f"--- INSPECTING: {path} ---")
    try:
        with open(path, "rb") as f:
            state = pickle.load(f)
            
        print(f"Generation: {state.get('generation')}")
        print(f"Population Size: {len(state.get('population', []))}")
        
        best = state.get('best_genome')
        if best:
            print(f"Best Genome ID: {best.id}")
            print(f"Best Fitness: {best.fitness}")
            print(f"Best Params: {best.params}")
            
        print("\nKeys in checkpoint:", state.keys())
        print("----------------------------")
        
    except ModuleNotFoundError as e:
        print(f"ERROR: Could not load objects. Missing module: {e}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    # Change this to whatever file you want to check
    inspect("test_data/complex_test_gen_0.pkl")