import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
import io
import os
from PIL import Image

# ==========================================
# 1. CONFIGURAÇÃO DOS PARÂMETROS
# ==========================================
@dataclass
class CircuitParameters:
    # --- Parâmetros Físicos ---
    inductance_h: float = 1e-3      # 1 mH
    resistance_ohm: float = 1e-6    # 1 microOhm (ramo da chave)
    capacitance_f: float = 10e-9    # 10 nF
    frequency_hz: float = 60.0      # 60 Hz
    voltage_peak_v: float = 10000.0 # 10 kV
    phase_deg: float = 0.0           
    
    # --- Parâmetros de Simulação e GIF ---
    cycles_before: float = 0.25      # Tempo antes da abertura (em ciclos)
    ms_after: float = 4.0            # Tempo após a abertura (em milissegundos)
    sampling_freq_hz: float = 1e6    # Frequência de Amostragem Constante (1 GHz)
    gif_total_frames: int = 60       # Quantidade de quadros na animação

# ==========================================
# 2. MOTOR DE SIMULAÇÃO
# ==========================================
class CircuitSimulator:
    def __init__(self, data: CircuitParameters):
        self.data = data
        self.results = {}

    def run_simulation(self):
        p = self.data
        w = 2 * np.pi * p.frequency_hz
        phi_v = np.deg2rad(p.phase_deg)
        dt = 1.0 / p.sampling_freq_hz
        
        # Cálculo de Regime Estacionário (Chave Fechada)
        ZL = 1j * w * p.inductance_h
        ZC = 1 / (1j * w * p.capacitance_f)
        R = p.resistance_ohm
        Z_para = (R * ZC) / (R + ZC)
        Z_total = ZL + Z_para
        
        V_comp = p.voltage_peak_v * np.exp(1j * phi_v)
        I_L_comp = V_comp / Z_total
        V_C_comp = I_L_comp * Z_para
        I_R_comp = V_C_comp / R
        
        # Encontrar instante de abertura (Zero da corrente ir)
        phase_ir = np.angle(I_R_comp)
        k = np.ceil(phase_ir / np.pi)
        t_open = (k * np.pi - phase_ir) / w
        if t_open < 1e-10: t_open += np.pi / w
        
        # Vetores de Tempo (Amostragem Constante)
        t_dur_pre = p.cycles_before * (1.0 / p.frequency_hz)
        t_pre = np.linspace(t_open - t_dur_pre, t_open, int(t_dur_pre / dt))
        
        t_dur_post = p.ms_after / 1000.0
        t_post = np.linspace(t_open, t_open + t_dur_post, int(t_dur_post / dt))[1:]
        dt_post = t_post - t_open
        
        # Cálculo das Ondas
        vs_pre = p.voltage_peak_v * np.sin(w * t_pre + phi_v)
        il_pre = np.abs(I_L_comp) * np.sin(w * t_pre + np.angle(I_L_comp))
        ir_pre = np.abs(I_R_comp) * np.sin(w * t_pre + phase_ir)
        vc_pre = np.abs(V_C_comp) * np.sin(w * t_pre + np.angle(V_C_comp))
        
        # Transiente Pós-Abertura (Circuito LC Série)
        w0 = 1 / np.sqrt(p.inductance_h * p.capacitance_f)
        IL0 = np.abs(I_L_comp) * np.sin(w * t_open + np.angle(I_L_comp))
        VC0 = np.abs(V_C_comp) * np.sin(w * t_open + np.angle(V_C_comp))
        VS0 = p.voltage_peak_v * np.sin(w * t_open + phi_v)
        
        il_post = IL0 * np.cos(w0 * dt_post) + (VS0 - VC0)/(w0 * p.inductance_h) * np.sin(w0 * dt_post)
        vc_post = VS0 - (VS0 - VC0)*np.cos(w0 * dt_post) + (IL0/(w0 * p.capacitance_f))*np.sin(w0 * dt_post)
        vs_post = p.voltage_peak_v * np.sin(w * t_post + phi_v)
        
        # Compilação dos Resultados
        self.results = {
            't': np.concatenate([t_pre, t_post]) * 1000, 
            'vs': np.concatenate([vs_pre, vs_post]),
            'il': np.concatenate([il_pre, il_post]),
            'ir': np.concatenate([ir_pre, np.zeros_like(t_post)]),
            'vc': np.concatenate([vc_pre, vc_post]),
            'vsw': np.concatenate([np.zeros_like(t_pre), vc_post]),
            't_open_ms': t_open * 1000
        }

# ==========================================
# 3. FUNÇÕES DE VISUALIZAÇÃO
# ==========================================
def create_frame(sim, end_idx, show_legend=True):
    res = sim.results
    p = sim.data
    v_lim = p.voltage_peak_v * 2.1
    i_lim = np.max(np.abs(res['il'])) * 1.2
    
    # Padronização de Cores e Transparência
    colors = {'vs': '#1f77b4', 'vc': '#ff7f0e', 'il': '#9467bd', 'ir': '#8c564b', 'vsw': '#d62728'}
    op = 0.65

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("Tensões no Sistema (V)", "Correntes nos Ramos (A)", "Foco na Interrupção: Tensão e Corrente na Chave"),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}]]
    )

    t_c = res['t'][:end_idx]

    # Subplot 1: Tensões
    fig.add_trace(go.Scatter(x=t_c, y=res['vs'][:end_idx], name="V Fonte", line=dict(color=colors['vs']), opacity=op, legendgroup="V Fonte"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_c, y=res['vc'][:end_idx], name="V Capacitor", line=dict(color=colors['vc']), opacity=op, legendgroup="V Cap"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_c, y=res['vsw'][:end_idx], name="V Chave (TRV)", line=dict(color=colors['vsw']), opacity=op, legendgroup="V Chave"), row=1, col=1)

    # Subplot 2: Correntes
    fig.add_trace(go.Scatter(x=t_c, y=res['il'][:end_idx], name="i Indutor", line=dict(color=colors['il']), opacity=op, legendgroup="i Indutor"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_c, y=res['ir'][:end_idx], name="i Chave", line=dict(color=colors['ir']), opacity=op, legendgroup="i Chave"), row=2, col=1)

    # Subplot 3: Foco Unificado (Apenas Vsw e Ir)
    fig.add_trace(go.Scatter(x=t_c, y=res['vsw'][:end_idx], name="V Chave (TRV)", line=dict(color=colors['vsw']), opacity=op, showlegend=False, legendgroup="V Chave"), row=3, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=t_c, y=res['ir'][:end_idx], name="i Chave", line=dict(color=colors['ir']), opacity=op, showlegend=False, legendgroup="i Chave"), row=3, col=1, secondary_y=True)

    # Fixação de Escalas e Estética
    for r in [1, 2, 3]:
        fig.update_xaxes(range=[res['t'][0], res['t'][-1]], row=r, col=1)
    
    fig.update_yaxes(range=[-v_lim, v_lim], row=1, col=1)
    fig.update_yaxes(range=[-i_lim, i_lim], row=2, col=1)
    fig.update_yaxes(range=[-v_lim, v_lim], row=3, col=1, secondary_y=False)
    fig.update_yaxes(range=[-i_lim, i_lim], row=3, col=1, secondary_y=True)

    fig.add_vline(x=res['t_open_ms'], line_dash="dash", line_color="black", opacity=0.3)
    fig.update_layout(template="plotly_white", showlegend=show_legend, height=900, width=1000,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
    
    return fig

# ==========================================
# 4. EXECUÇÃO E EXPORTAÇÃO
# ==========================================
if __name__ == "__main__":
    # 1. Simulação
    params = CircuitParameters()
    sim = CircuitSimulator(params)
    sim.run_simulation()
    
    # 2. Exibir gráfico interativo no Navegador
    print("Abrindo gráficos interativos no navegador...")
    fig_final = create_frame(sim, len(sim.results['t']), show_legend=True)
    fig_final.show()
    
    # 3. Gerar GIF Animado
    print(f"Iniciando geração do GIF ({params.gif_total_frames} frames)...")
    frames_img = []
    indices = np.linspace(5, len(sim.results['t']), params.gif_total_frames, dtype=int)
    
    for idx in indices:
        fig = create_frame(sim, idx, show_legend=True)
        # scale=1 para manter o tamanho de arquivo aceitável
        img_bytes = fig.to_image(format="png", width=1000, height=1000, scale=1, engine="kaleido")
        frames_img.append(Image.open(io.BytesIO(img_bytes)))
    
    # Definição de durações (200ms por frame e 1.5s de pausa no final)
    durs = [200] * (len(frames_img) - 1) + [1500] 

    output_name = "trv_final_animation.gif"
    frames_img[0].save(output_name, save_all=True, append_images=frames_img[1:], duration=durs, loop=0)
    
    print(f"--- PROCESSO CONCLUÍDO ---")
    print(f"Arquivo gerado: {os.path.abspath(output_name)}")