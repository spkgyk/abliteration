import torch
import einops
import time

# Set the seed for reproducibility
torch.manual_seed(0)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample tensor dimensions
bs = 32
S = 400
d_model = 2048

# Generate random tensors for activation and refusal_direction
activation = torch.randn(bs, S, d_model, device=device)
refusal_direction = torch.randn(d_model, device=device)
refusal_direction = refusal_direction / torch.norm(refusal_direction)  # Normalize


# Function for direction ablation using torch.matmul
def direction_ablation_matmul(activation, refusal_direction):
    proj = activation @ refusal_direction.unsqueeze(-1) * refusal_direction
    return activation - proj


# Function for direction ablation using torch.einsum
def direction_ablation_torch_einsum(activation, refusal_direction):
    proj = torch.einsum("...d,ds->...s", activation, refusal_direction.view(-1, 1)) * refusal_direction
    return activation - proj


# Function for direction ablation using einops.einsum
def direction_ablation_einops(activation, refusal_direction):
    proj = einops.einsum(activation, refusal_direction.view(-1, 1), "... d_act, d_act single -> ... single") * refusal_direction
    return activation - proj


# New functions for matrix orthogonalization
def get_orthogonalized_matrix_einops(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    proj = einops.einsum(matrix, vec, "d_model n, d_model -> n")
    result = matrix - einops.einsum(vec, proj, "d_model, n -> d_model n")
    return result


def get_orthogonalized_matrix_matmul(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    # Compute the projection of vec onto each column of the matrix
    proj = torch.matmul(vec, matrix)

    # Outer product of vec and proj using torch.outer
    outer_product = torch.outer(vec, proj)

    # Subtract the projection from the original matrix
    result = matrix - outer_product

    return result


# Warm up and check results
output_matmul = direction_ablation_matmul(activation, refusal_direction)
output_torch_einsum = direction_ablation_torch_einsum(activation, refusal_direction)
output_einops_einsum = direction_ablation_einops(activation, refusal_direction)

# Ensure all outputs are the same
assert torch.allclose(output_matmul, output_torch_einsum), "torch.matmul and torch.einsum outputs differ"
assert torch.allclose(output_matmul, output_einops_einsum), "torch.matmul and einops.einsum outputs differ"
assert torch.allclose(output_torch_einsum, output_einops_einsum), "torch.einsum and einops.einsum outputs differ"

# Generate random matrix for orthogonalization tests
matrix = torch.randn(d_model, S, device=device)
vec = torch.randn(d_model, device=device)
vec = vec / torch.norm(vec)  # Normalize

# Warm up and check results for matrix orthogonalization
output_einops = get_orthogonalized_matrix_einops(matrix, vec)
output_matmul = get_orthogonalized_matrix_matmul(matrix, vec)

# Ensure outputs are the same
assert torch.allclose(output_einops, output_matmul), "einops and matmul orthogonalization outputs differ"


# Function to measure execution time
def measure_execution_time(func, *args, n_iterations=5000):
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_iterations):
        # result = func(*args)
        torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time


# Measure execution times
einsum_torch_time = measure_execution_time(direction_ablation_torch_einsum, activation, refusal_direction)
einsum_einops_time = measure_execution_time(direction_ablation_einops, activation, refusal_direction)
matmul_time = measure_execution_time(direction_ablation_matmul, activation, refusal_direction)
ortho_einops_time = measure_execution_time(get_orthogonalized_matrix_einops, matrix, vec)
ortho_matmul_time = measure_execution_time(get_orthogonalized_matrix_matmul, matrix, vec)

print("Direction Ablation:")
print(f"Execution time for torch.matmul: {matmul_time:.6f} seconds")
print(f"Execution time for torch.einsum: {einsum_torch_time:.6f} seconds")
print(f"Execution time for einops.einsum: {einsum_einops_time:.6f} seconds")

print("\nMatrix Orthogonalization:")
print(f"Execution time for einops version: {ortho_einops_time:.6f} seconds")
print(f"Execution time for matmul version: {ortho_matmul_time:.6f} seconds")

# Clear GPU memory
torch.cuda.empty_cache()
