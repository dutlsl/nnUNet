# Agent Rules for nnUNet Workspace

This workspace uses a specific Python virtual environment for all development, training, and testing tasks.

## Virtual Environment Details
- **Python**: 3.10

## Rules for Command Execution
1. **Always use the virtual environment**: When running any commands (e.g., `python`, `pip`, `pytest`, or `nnUNetv2_*` entry points), you must activate the environment first or use the absolute path to the virtual environment's python.
   - **Activation**:
     ```bash
     source /home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/activate
     ```
   - **Direct Execution**: Use `/home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/python` instead of system `python`.

2. **Environment Verification Harness**:
   - To verify the integrity of the setup (including PyTorch, CUDA support, and nnUNetv2 imports), run:
     ```bash
     /home/iulab0/PycharmProjects/nnUNet/nnunet_env/bin/python /home/iulab0/PycharmProjects/nnUNet/agent/test_dummy_nnunet.py
     ```

3. **Long-Running Process Rule (Mandatory)**:
   - For ANY command or training/evaluation task with an expected execution time of **1 hour or longer**, NEVER run it as a standard session-dependent background process.
   - You MUST run it as a detached process using `nohup` (e.g., `nohup <command> > <log_file> 2>&1 &`) so that it survives IDE restarts, agent session terminations, or terminal disconnections.

