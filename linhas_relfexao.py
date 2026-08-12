import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# --- Configurações Globais de Estética ---
plt.rcParams.update({'font.size': 14})

# --- Parâmetros de Simulação ---
L = 1.0  # Comprimento unitário [m]
v = 1.0  # Velocidade de propagação [m/s]
T_total = 50.0  # Tempo total de simulação [s]
Nx = 2 ** 10 # Resolução espacial (1024 pontos)
dx = L / (Nx - 1)
dt = 0.9 * (dx / v)  # Condição de Estabilidade CFL
Nt = int(T_total / dt)


def simulate(gamma_type, input_type):
    """Executa a simulação FDTD para um dado Gamma e tipo de entrada."""
    V = np.zeros(Nx)
    I = np.zeros(Nx - 1)

    # Históricos para animação (Espaço, Tempo, Fonte, Carga)
    h_V, h_t, h_x0, h_xL = [], [], [], []

    for n in range(Nt):
        t = n * dt

        # 1. Atualiza Corrente (I)
        I -= (dt / dx) * (V[1:] - V[:-1])

        # 2. Atualiza Tensão/Torque (V)
        V[1:-1] -= (dt / dx) * (I[1:] - I[:-1])

        # 3. Aplica Entrada na Fonte (x=0)
        if input_type == 'step':
            V[0] = 1.0
        elif input_type == 'pulse':
            # Pulso Gaussiano estreito (sigma=0.005)
            V[0] = np.exp(-((t - 0.3) ** 2) / (2 * 0.01 ** 2))
        elif input_type == 'sine':
            V[0] = np.sin(2 * np.pi * 0.5 * t)

        # 4. Condições de Contorno na Carga (x=L)
        if gamma_type == 1:  # Aberto / Ponta Fixa
            V[-1] -= (dt / dx) * (0 - I[-1])
        elif gamma_type == -1:  # Curto / Ponta Livre
            V[-1] = 0
        elif gamma_type == 0:  # Casado / Absorção
            V[-1] = V[-1] - (v * dt / dx) * (V[-1] - V[-2])

        # Grava frames a cada 20 iterações para manter performance
        if n % 20 == 0:
            h_V.append(V.copy())
            h_t.append(t)
            h_x0.append(V[0])
            h_xL.append(V[-1])

    return h_V, h_t, h_x0, h_xL


def create_video(input_type, filename, title_prefix):
    """Gera o vídeo com a grade de 3x3 gráficos."""
    print(f"Iniciando simulações para: {title_prefix}...")
    results = [simulate(1, input_type), simulate(-1, input_type), simulate(0, input_type)]

    titles = [r"$\Gamma = 1$ (Fixed / Open)", r"$\Gamma = -1$ (Free / Short)", r"$\Gamma = 0$ (Matched)"]

    # Figura grande para acomodar todos os labels de eixos
    fig = plt.figure(figsize=(26, 16))
    fig.suptitle(f"Analysis Group: Response to {title_prefix}", fontsize=24, fontweight='bold', y=0.96)

    # hspace aumentado para 0.7 para evitar sobreposição de labels e títulos
    gs = fig.add_gridspec(3, 3, hspace=0.7, wspace=0.3)

    lines_space, lines_t0, lines_tL = [], [], []
    x_axis = np.linspace(0, L, Nx)

    for col in range(3):
        # --- Linha 0: Distribuição Espacial ---
        ax_s = fig.add_subplot(gs[0, col])
        l_s, = ax_s.plot(x_axis, results[col][0][0], lw=2, color='#0082c9')
        ax_s.set_xlim(0, L)
        ax_s.set_ylim(-2.5, 2.5)
        ax_s.set_title(titles[col], fontweight='bold', fontsize=20, pad=15)
        ax_s.set_xlabel('Position x [m]', fontsize=14)
        if col == 0: ax_s.set_ylabel('Space [p.u.]', fontweight='bold', fontsize=16)
        ax_s.grid(True, alpha=0.3)
        lines_space.append(l_s)

        # --- Linhas 1 e 2: Respostas Temporais (x=0 e x=L) ---
        for row, color, label, lines_list in zip([1, 2], ['#2ca02c', '#ff7f0e'],
                                                 ['$x=0$', '$x=L$'], [lines_t0, lines_tL]):
            ax = fig.add_subplot(gs[row, col])
            l, = ax.plot([], [], lw=2, color=color)
            ax.set_xlim(0, T_total)
            ax.set_ylim(-2.5, 2.5)
            ax.set_xlabel('Time [s]', fontsize=14)
            ax.grid(True, alpha=0.3)
            if col == 0: ax.set_ylabel(f'{label} [p.u.]', fontweight='bold', fontsize=16)
            lines_list.append(l)

    def animate(i):
        artists = []
        for col in range(3):
            h_V, h_t, h_x0, h_xL = results[col]

            # Atualiza gráfico de espaço
            lines_space[col].set_ydata(h_V[i])

            # Atualiza gráficos de tempo
            lines_t0[col].set_data(h_t[:i + 1], h_x0[:i + 1])
            lines_tL[col].set_data(h_t[:i + 1], h_xL[:i + 1])

            artists.extend([lines_space[col], lines_t0[col], lines_tL[col]])
        return artists

    # Criação da animação
    ani = plt.matplotlib.animation.FuncAnimation(fig, animate, frames=len(results[0][0]), interval=50, blit=True)

    # Salvamento (Requer FFmpeg instalado)
    print(f"Salvando vídeo: {filename}...")
    writer = FFMpegWriter(fps=20, bitrate=4000)
    ani.save(filename, writer=writer, dpi=150)
    plt.close(fig)
    print(f"Concluído: {filename}\n")


# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    try:
        # create_video('pulse', 'analise_impulso.mp4', 'Impulse')
        create_video('step', 'analise_degrau.mp4', 'Step')
        create_video('sine', 'analise_senoide.mp4', 'Sine Wave')
        print("Todos os vídeos do Grupo de Análise foram gerados com sucesso!")
    except Exception as e:
        print(f"Erro durante a execução: {e}")
        print("Dica: Verifique se o FFmpeg está instalado e no PATH do sistema.")