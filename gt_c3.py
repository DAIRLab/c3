import numpy as np
import matplotlib.pyplot as plt
def load_data(file_path, arr_len):
    chunks = []
    current_chunk = []
    with open(file_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "":
                # Skip empty line and start new chunk
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            else:
                current_chunk.append(float(stripped))

        # In case the file doesn't end with an empty line
        if current_chunk:
            chunks.append(current_chunk)

    data = np.array(chunks, dtype=np.float32)
    return data
        
def make_plot(arr_list, name_list):
    fig = plt.figure(figsize=(10, 6))
    for arr, name in zip(arr_list, name_list):

        plt.plot(arr, label=f"${name}$")
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.title("GT lambda and gamma")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        

delta_file = "/home/yufeiyang/Documents/c3/c3_delta_output.txt"
gamma_file = "/home/yufeiyang/Documents/c3/c3_gamma_output.txt"

delta_data = load_data(delta_file, 16)
delta_data = delta_data[:, 6:12]
gamma_data = load_data(gamma_file, 6)

print(delta_data.shape)
timesteps = np.arange(delta_data.shape[0])

# make_plot([delta_data[:, 0], gamma_data[:, 0]], ["G1 Maximum velocity", "gamma"])
# make_plot([delta_data[:, 1], gamma_data[:, 1]], ["G1 positive friction", "gamma"])
# make_plot([delta_data[:, 2], gamma_data[:, 2]], ["G1 negative friction", "gamma"])
# make_plot([delta_data[:, 3], gamma_data[:, 3]], ["G1Maximum velocity", "gamma"])
# make_plot([delta_data[:, 4], gamma_data[:, 4]], ["G2 positive friction", "gamma"])
# make_plot([delta_data[:, 5], gamma_data[:, 5]], ["G2 negative friction", "gamma"])
# plt.show()
# plt.figure(figsize=(10, 6))
# for i in range(6):
#     plt.plot(timesteps, delta_data[:, i], label=f"Signal {i+1}")

# plt.show()

def load_admm_data(file_path):
    blocks = []
    current_block = []

    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                # Empty line: end of block
                if current_block:
                    blocks.append(np.vstack(current_block))
                    current_block = []
            else:
                numbers = list(map(float, stripped.split()))
                current_block.append(numbers)
        
        # Catch any last block if file doesn't end with a blank line
        if current_block:
            blocks.append(np.vstack(current_block))

    # Convert to final 3D NumPy array: (N, 10, 16)
    data = np.stack(blocks)

    return data

def get_gamma(lambda_data):
    N = 10
    h = 0.1
    g = 9.81
    mu = 1.0
    dt = 0.01
    E = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 1, 0, -1, 0, 0],
                  [0, -1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, -1],
                  [0, -1, 0, 0, 0, 1]])
    F = np.array([[0, -1, -1, 0, 0, 0],
                  [1, 2 * h, -2 * h, 0, h, -h],
                  [1, -2 * h, 2 * h, 0, -h, h],
                  [0, 0, 0, 0, -1, -1],
                  [0, h, -h, 1, 2 * h, -2 * h],
                  [0, -h, h, 1, -2 * h, 2 * h]])
    d = np.array([[-g * h * h], [-g * h], [0], [0], [0], [0]])
    H = np.array([[0, 0, mu, 0], 
                  [-h, 0, 0, 0],
                  [h, 0, 0, 0],
                  [0, 0, 0, mu],
                  [0, -h, 0, 0],
                  [0, h, 0, 0]])
    x_part = lambda_data[:, :, 0:6]
    lambda_part = lambda_data[:, :, 6:12]
    u_part = lambda_data[:, :, 12:16]
    new_delta = np.zeros((lambda_part.shape[0], lambda_part.shape[1], 22))
    Ex = np.matmul(x_part, E.T)
    Fl = np.matmul(lambda_part, F.T)
    Hu = np.matmul(u_part, H.T)
    gamma = Ex + Fl + Hu + d.reshape(1, 1, -1)   
    new_delta[:, :, 0:6] = x_part
    new_delta[:, :, 6:12] = lambda_part
    new_delta[:, :, 12:16] = u_part
    new_delta[:, :, 16:22] = gamma
    return new_delta

def plot_x(file_path):
    pass
if __name__ == "__main__":
    # # load delta and z info
    delta = load_admm_data("/home/yufeiyang/Documents/c3/c3_debug_delta.txt") # 1990 10 16
    z = load_admm_data("/home/yufeiyang/Documents/c3/c3_debug_z.txt")
    print(delta.shape)
    print(z.shape)
    gamma_delta = get_gamma(delta)
    gamma_z = get_gamma(z)
    # save as npy file
    np.save("/home/yufeiyang/Documents/c3/debug_output/original_delta.npy", gamma_delta)
    np.save("/home/yufeiyang/Documents/c3/debug_output/original_z.npy", gamma_z)

    # x_data = np.loadtxt("/home/yufeiyang/Documents/c3/debug_output/original_c3_x.txt")
    # print(x_data.shape)
    # timesteps = np.arange(x_data.shape[0])  # shape: (200,)

    # # Plot each column as a separate line
    # labels = ["Object position", "Object velocity", "Gripper 1 position",
    #           "Gripper 1 velocity", "Gripper 2 position", "Gripper 2 velocity"]
    # plt.plot(timesteps, x_data[:, 0], label="Object position")
    # plt.plot(timesteps, x_data[:, 2], label="Gripper 1 position")
    # plt.plot(timesteps, x_data[:, 4], label="Gripper 2 position")

    # plt.xlabel("Time Step")
    # plt.ylabel("Value")
    # plt.title("Original C3 states")
    # plt.legend(fontsize=13)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()