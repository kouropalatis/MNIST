import sys
import torch
from mnist.model import Decoder, Encoder, Model

torch.serialization.add_safe_globals([Model, Encoder, Decoder])

if __name__ == "__main__":
    # Check if paths are provided
    if len(sys.argv) < 3:
        print("Usage: python reproducibility_tester.py <path_to_run1> <path_to_run2>")
        sys.exit(1)

    exp1 = sys.argv[1]
    exp2 = sys.argv[2]

    print(f"Comparing run {exp1} to {exp2}")

    # Load the whole model objects
    model1 = torch.load(f"{exp1}/trained_model.pt", map_location="cpu", weights_only=False)
    model2 = torch.load(f"{exp2}/trained_model.pt", map_location="cpu", weights_only=False)

    # Compare parameters one by one
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1, p2):
            msg = "encountered a difference in parameters, your script is not fully reproducible"
            raise RuntimeError(msg)
            
    print("✅ Success! The models are identical. Your script is fully reproducible.")