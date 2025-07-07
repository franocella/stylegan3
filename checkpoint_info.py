# inspect_checkpoint.py
import pickle
import torch
import sys

def inspect_stylegan3_pkl(file_path):
    """
    Loads a StyleGAN3 .pkl file and prints its generator's architecture parameters.
    """
    print(f"Inspecting checkpoint: {file_path}")
    
    try:
        # Load the pickle file using torch to ensure compatibility with GPU tensors
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # The generator model is usually stored in 'G_ema'
        if 'G_ema' not in data:
            raise KeyError("Could not find a generator model ('G_ema') in the checkpoint file.")

        G = data['G_ema']

        # --- Extract and print parameters ---
        print("\n## Detected Configuration ##")

        # Basic parameters
        print(f"  Image Resolution:   {G.img_resolution}")
        print(f"  Image Channels:     {G.img_channels}")
        
        # Latent space dimensions
        print(f"  Z Latent Dimension:   {G.z_dim}")
        print(f"  W Latent Dimension:   {G.w_dim}")

        # Conditional parameter
        is_conditional = G.c_dim > 0
        print(f"  Conditional Model:    {is_conditional}")
        if is_conditional:
            print(f"  Label Dimension c:      {G.c_dim}")

        # Mapping network parameters
        if hasattr(G, 'mapping') and hasattr(G.mapping, 'num_layers'):
            print(f"  Mapping Depth (map-depth): {G.mapping.num_layers}")

        # Synthesis network parameters
        if hasattr(G, 'synthesis') and hasattr(G.synthesis, 'channel_base'):
            print(f"  Channel Base (channel_base): {G.synthesis.channel_base}")
        
        if hasattr(G, 'synthesis') and hasattr(G.synthesis, 'channel_max'):
            print(f"  Max Channels (cmax):         {G.synthesis.channel_max}")

        print("\nNOTE: The '--cfg' parameter (e.g., 'stylegan3-t') is not explicitly saved,")
        print("but can be inferred from this combination of parameters.")

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except (pickle.UnpicklingError, KeyError, AttributeError) as e:
        print(f"Error: The file does not appear to be a valid StyleGAN3 checkpoint or is corrupted. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoint.py <path_to_your_checkpoint.pkl>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    inspect_stylegan3_pkl(checkpoint_path)