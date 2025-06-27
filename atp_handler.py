import os
import subprocess
import scipy.io as sio
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class ATPHandler:
    def __init__(self, tpbigG_exe_phat, pl42mat_exe_phat):
        """
        This class encapsulates methods for ATP simulation,
        modifying CSYS parameters, and plotting dataframes.

        Parameters:
        tpbigG_exe_phat (str): Path to the directory where tpbigG.exe is located.
        pl42mat_exe_phat (str): Path to the Pl42mat executable.
        """
        self.tpbigG_exe_phat = tpbigG_exe_phat
        self.pl42mat_exe_phat = pl42mat_exe_phat

    def run_atp_simulation(self, atp_file_path):
        """
        Executes the ATP simulation using tpbigG.exe and Pl42mat,
        and then loads the resulting .mat file as a pandas DataFrame.

        Parameters:
        atp_file_path (str): Full path to the .atp file to be executed.

        Returns:
        pd.DataFrame: Filtered data as a pandas DataFrame.
        """
        # Change the working directory to the root project path
        os.chdir(self.tpbigG_exe_phat)

        # Run the ATP simulation
        command = fr"tpbigG.exe {atp_file_path}"
        subprocess.run(command, shell=True)

        # Convert .pl4 file to .mat using Pl42mat
        pl4_file_path = atp_file_path.replace(".atp", ".pl4")
        pl42mat_command = fr'"{self.pl42mat_exe_phat} {pl4_file_path}"'
        os.system(pl42mat_command)

        # Load .mat data
        mat_file_path = atp_file_path.replace(".atp", ".mat")
        print(mat_file_path)
        mat_data = sio.loadmat(mat_file_path)

        # Filter and transform into DataFrame
        filtered_data_dic = {
            key: mat_data[key][3:].flatten() for key in mat_data.keys() if not key.startswith('__')
        }
        filtered_data_df = pd.DataFrame(filtered_data_dic)

        return filtered_data_df

    def alterar_valores_csys(self, atp_file_path, lista_csys):
        """
        Modifies the 'CSYS' parameter in the .atp file and saves a new file
        for each value in the list.

        Parameters:
        atp_file_path (str): Path to the original .atp file.
        lista_csys (list): List of new values for 'CSYS'.

        Returns:
        list: List of paths of the saved new files.
        """
        novos_arquivos = []

        # Loop through the list of CSYS values
        for csys in lista_csys:
            # Read original file
            with open(atp_file_path, 'r', encoding='latin-1') as file:
                content = file.readlines()

            # Search and replace the line that contains "CSYS"
            for i, linha in enumerate(content):
                if "CSYS" in linha:
                    content[i] = f"CSYS ={csys}. $$\n"
                    break

            # Set the name of the new file
            output_file_path = atp_file_path.replace(".atp", f"_CSYS_{csys}.atp")

            # Save the modified file
            with open(output_file_path, 'w', encoding='latin-1') as file:
                file.writelines(content)

            # Store in the list
            novos_arquivos.append(output_file_path)

        return novos_arquivos

    def alterar_valores_C69(self, atp_file_path: str, C69_list: list[float]) -> list[str]:
        """
        Create one .atp copy for each value in *C69_list*, replacing the shunt-
        capacitor connected to node XX69-ground.

        Parameters
        ----------
        atp_file_path : str
            Path to original .atp file.
        C69_list : list[float]
            Capacitance values (μF) to insert.

        Returns
        -------
        list[str]
            Paths of the new .atp files.
        """
        novos_arquivos = []

        for c69 in C69_list:
            with open(atp_file_path, "r", encoding="latin-1") as f:
                linhas = f.readlines()

            for i, linha in enumerate(linhas):
                # The shunt branch line starts with 'XX69' and has at least 3 numbers: R  C  flag
                if linha.lstrip().startswith("XX69"):
                    tokens = linha.split()
                    if len(tokens) >= 3:           # tokens[2] is the C value
                        tokens[2] = f"{c69}"        # replace capacitance
                        linhas[i] = "  " + " ".join(tokens) + "\n"
                        break

            novo_nome = atp_file_path.replace(".atp", f"_C69_{c69}.atp")
            with open(novo_nome, "w", encoding="latin-1") as f:
                f.writelines(linhas)

            novos_arquivos.append(novo_nome)

        return novos_arquivos


    def plot_multiple_dataframes(self, data_for_plot, title, names, f1):
        """
        Plots multiple DataFrames in a single figure using Plotly.

        Parameters:
        data_for_plot (list): List of DataFrames to be plotted.
        title (str): Title of the plot.
        names (list): Names for each trace in the plot.

        Returns:
        go.Figure: Plotly Figure object.
        """
        fig = go.Figure()

        # Add each DataFrame to the figure
        for i, df in enumerate(data_for_plot):
            name = names[i]
            fig.add_trace(
                go.Scatter(
                    x=df.iloc[:, 0]/f1,
                    y=df.iloc[:, 1],
                    mode='lines',
                    name=name
                )
            )

        # Customize layout
        fig.update_layout(
            title=title,
            showlegend=True,
            template='plotly_white',
            height=600,
            width=1200,
            yaxis=dict(
                type='log')
        )

        return fig