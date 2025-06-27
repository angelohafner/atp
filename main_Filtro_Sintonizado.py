import numpy as np
from atp_handler import ATPHandler

if __name__ == "__main__":
    tpbigG_exe_phat = r"C:\ATP\ATP\GNUATP"
    pl42mat_exe_phat = r"C:\ATP\Pl42mat09\Pl42mat"
    atp_file_path = r"C:\ATP\DHP\curso_pscad\banco_curso_pscad_1.atp"

    C69_list = [1, 25, 50, 75]           # μF
    nomes = [f"C69_{c}" for c in C69_list]

    atp = ATPHandler(tpbigG_exe_phat, pl42mat_exe_phat)
    arquivos_mod = atp.alterar_valores_C69(atp_file_path, C69_list)

    dados = [atp.run_atp_simulation(a) for a in arquivos_mod]
    fig = atp.plot_multiple_dataframes(dados, "", nomes, f1=60)
    fig.show()
