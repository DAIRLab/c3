import matplotlib.pyplot as plt
import numpy as np

def dual_twinx(delta_sol, label, original=False):
    fig, ax_left = plt.subplots(6, 1, figsize=(12, 18), constrained_layout=True)
    ax_right = [ax.twinx() for ax in ax_left]  # Create right y-axes

    system_iter = 300
    dt = 0.01
    if original:
        delta_sol = delta_sol
    else:
        delta_sol = delta_sol[:, 0, :]

    # Gripper 1
    ax_left[0].plot(delta_sol[:, 6], 'tab:blue', label="Gripper 1 Slack var")
    ax_right[0].plot(delta_sol[:, 16], 'tab:red', label=r"$\gamma_1$")
    ax_left[0].set_ylabel("Slack var")
    ax_right[0].set_ylabel(r"$\gamma_1$")
    ax_left[0].legend(loc="upper left")
    ax_right[0].legend(loc="upper right")

    ax_left[1].plot(delta_sol[:, 7], 'tab:blue', label="Gripper 1 Pos tangential force")
    ax_right[1].plot(delta_sol[:, 17], 'tab:red', label=r"$\gamma_1$")
    ax_left[1].set_ylabel("Pos tangential force")
    ax_right[1].set_ylabel(r"$\gamma_1$")
    ax_left[1].legend(loc="upper left")
    ax_right[1].legend(loc="upper right")

    ax_left[2].plot(delta_sol[:, 8], 'tab:blue', label="Gripper 1 Neg tangential force")
    ax_right[2].plot(delta_sol[:, 18], 'tab:red', label=r"$\gamma_1$")
    ax_left[2].set_ylabel("Neg tangential force")
    ax_right[2].set_ylabel(r"$\gamma_1$")
    ax_left[2].legend(loc="upper left")
    ax_right[2].legend(loc="upper right")

    # Gripper 2
    ax_left[3].plot(delta_sol[:, 9], 'tab:blue', label="Gripper 2 Slack var")
    ax_right[3].plot(delta_sol[:, 19], 'tab:red', label=r"$\gamma_2$")
    ax_left[3].set_ylabel("Slack var")
    ax_right[3].set_ylabel(r"$\gamma_2$")
    ax_left[3].legend(loc="upper left")
    ax_right[3].legend(loc="upper right")

    ax_left[4].plot(delta_sol[:, 10], 'tab:blue', label="Gripper 2 Pos tangential force")
    ax_right[4].plot(delta_sol[:, 20], 'tab:red', label=r"$\gamma_2$")
    ax_left[4].set_ylabel("Pos tangential force")
    ax_right[4].set_ylabel(r"$\gamma_2$")
    ax_left[4].legend(loc="upper left")
    ax_right[4].legend(loc="upper right")

    ax_left[5].plot(delta_sol[:, 11], 'tab:blue', label="Gripper 2 Neg tangential force")
    ax_right[5].plot(delta_sol[:, 21], 'tab:red', label=r"$\gamma_2$")
    ax_left[5].set_ylabel("Neg tangential force")
    ax_right[5].set_ylabel(r"$\gamma_2$")
    ax_left[5].legend(loc="upper left")
    ax_right[5].legend(loc="upper right")

    fig.suptitle(label)
    plt.tight_layout()

def read_text(file_path):
    blocks = []
    current_block = []

    with open(file_path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped == "":
                # Empty line: end of a block
                if current_block:
                    blocks.append(np.array(current_block, dtype=float))
                    current_block = []
            else:
                current_block.append(float(stripped))

        # Don't forget the last block
        if current_block:
            blocks.append(np.array(current_block, dtype=float))
    blocks = np.array(blocks)
    return blocks

if __name__ == "__main__":
    # Load the delta_sol data
    delta_sol = np.load('/home/yufeiyang/Documents/c3/debug_output/delta_sol.npy')
    # Check the shape of delta_sol
    print(f"delta_sol shape: {delta_sol.shape}")
    
    # Call the dual_twinx function to plot
    dual_twinx(delta_sol, "Improved C3 Delta Solution")
    

    c3_delta = read_text('/home/yufeiyang/Documents/c3/c3_delta_output.txt')
    c3_gamma = read_text('/home/yufeiyang/Documents/c3/c3_gamma_output.txt')
    original_delta_sol = np.hstack((c3_delta, c3_gamma))

    dual_twinx(original_delta_sol, "Original C3 Delta Solution", True)

    print(f"c3_delta shape: {c3_delta.shape}")
    print(f"c3_gamma shape: {c3_gamma.shape}")
    print(f"original_delta_sol shape: {original_delta_sol.shape}")

        # Show the plot
    plt.show()
