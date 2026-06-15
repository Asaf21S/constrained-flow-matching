import os
import ast


def print_project_tree(directory, ignore_dirs=None):
    if ignore_dirs is None:
        # Ignore virtual environments, git, and cache folders
        ignore_dirs = {'.git', '__pycache__', 'venv', 'env', '.ipynb_checkpoints', 'node_modules'}

    print(f"📂 {os.path.basename(os.path.abspath(directory))}/")

    for root, dirs, files in os.walk(directory):
        # Filter out ignored directories in-place
        dirs[:] = [d for d in sorted(dirs) if d not in ignore_dirs]

        level = root.replace(directory, '').count(os.sep)
        if level > 0:
            indent = ' ' * 4 * level
            print(f"{indent}📂 {os.path.basename(root)}/")

        subindent = ' ' * 4 * (level + 1)
        for f in sorted(files):
            print(f"{subindent}📄 {f}")

            # If it's a Python file, look inside it
            if f.endswith('.py'):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        tree = ast.parse(file.read(), filename=filepath)

                        for node in tree.body:
                            # Catch standalone functions
                            if isinstance(node, ast.FunctionDef):
                                print(f"{subindent}    ⚙️ def {node.name}()")

                            # Catch classes and their internal methods
                            elif isinstance(node, ast.ClassDef):
                                print(f"{subindent}    📦 class {node.name}:")
                                for subnode in node.body:
                                    if isinstance(subnode, ast.FunctionDef):
                                        print(f"{subindent}        ⚙️ def {subnode.name}()")
                except Exception as e:
                    print(f"{subindent}    ⚠️ [Error parsing file: {e}]")


# Run it on the current directory
if __name__ == "__main__":
    print_project_tree('.')