import os
import sys
import io
import base64
import contextlib
import nbformat as nbf
import matplotlib
matplotlib.use('Agg')  # Non-blocking headless backend
import matplotlib.pyplot as plt

def execute_notebook(nb_path):
    print(f"Executing notebook: {nb_path}...")
    sys.stdout.flush()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    # Prepare shared execution namespace
    global_ns = {
        'display': lambda x: print(x.to_string() if hasattr(x, 'to_string') else str(x))
    }

    # Pre-populate matplotlib in namespace to prevent blocking plt.show()
    exec("import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt", global_ns)
    global_ns['plt'].show = lambda *args, **kwargs: None  # No-op show to avoid figure clearance

    executed_cells = 0
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue

        executed_cells += 1
        print(f"  Executing code cell {executed_cells} of 12...")
        sys.stdout.flush()
        cell.outputs = []
        cell.execution_count = executed_cells

        # Capture stdout
        f_stdout = io.StringIO()
        global_ns['plt'].close('all')

        try:
            with contextlib.redirect_stdout(f_stdout):
                exec(cell.source, global_ns)
        except Exception as e:
            print(f"  Error in cell {executed_cells}: {e}")
            sys.stdout.flush()
            err_output = nbf.v4.new_output(
                output_type='error',
                ename=type(e).__name__,
                evalue=str(e),
                traceback=[str(e)]
            )
            cell.outputs.append(err_output)
            continue

        stdout_text = f_stdout.getvalue()
        if stdout_text.strip():
            cell.outputs.append(nbf.v4.new_output(output_type='stream', name='stdout', text=stdout_text))

        # Capture any matplotlib figures created in this cell
        fig_nums = global_ns['plt'].get_fignums()
        for fignum in fig_nums:
            fig = global_ns['plt'].figure(fignum)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', dpi=130, bbox_inches='tight')
            img_buf.seek(0)
            b64_data = base64.b64encode(img_buf.read()).decode('utf-8')
            img_output = nbf.v4.new_output(
                output_type='display_data',
                data={'image/png': b64_data}
            )
            cell.outputs.append(img_output)
        global_ns['plt'].close('all')

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"SUCCESS: Notebook executed with all outputs and figures embedded: {nb_path}")
    sys.stdout.flush()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nb_p1 = os.path.join(base_dir, "AUV_Full_EDA_Analysis.ipynb")
    execute_notebook(nb_p1)

    # Mirror executed notebook to root
    root_nb = os.path.join(os.path.dirname(base_dir), "AUV_Full_EDA_Analysis.ipynb")
    import shutil
    shutil.copyfile(nb_p1, root_nb)
    print(f"Mirrored executed notebook to root: {root_nb}")
