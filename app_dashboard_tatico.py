"""
============================================================
Projeto CARTIO
Autoria: Wagner Calazans
Ano de criação: 2026
Versao: 5.0 (Cinemática Automatizada e Storytelling)
IME - Instituto Militar de Engenharia
Arquivo: app_dashboard_tatico.py
============================================================
"""

import streamlit as st
import networkx as nx
import osmnx as ox
import pydeck as pdk
import math
import numpy as np
import time

# --- Configurações da Interface Web ---
st.set_page_config(
    page_title="Simulador Tático - CARTIO", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Gerenciamento de Estado da Demo ---
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if "progresso_missao" not in st.session_state:
    st.session_state.progresso_missao = 0.0

is_demo = st.session_state.demo_mode

# --- Estilização CSS ---
estilo_css = """
<style>
    .header-container {
        display: flex;
        align-items: center;
        padding: 1rem 1rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border-bottom: 3px solid #861211;
        background-color: rgba(43, 117, 116, 0.1);
    }
    .header-title { display: flex; flex-direction: column; }
    .header-title h1 { margin: 0; font-size: 1.8rem; font-weight: 600; letter-spacing: 1px; }
    .header-title span { font-size: 0.9rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 2px; color: #2B7574; }
    .status-online { background-color: #2B7574; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9rem; font-weight: bold; }
    .status-offline { background-color: #861211; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.9rem; font-weight: bold; }
</style>
"""
st.markdown(estilo_css, unsafe_allow_html=True)

# --- Cabeçalho Customizado ---
svg_ime_logo = """<svg style="width:60px; height:60px; margin-right:1.5rem;" viewBox="0 0 100 120" xmlns="http://www.w3.org/2000/svg">
<path d="M10,10 L90,10 L90,60 C90,90 50,110 50,110 C50,110 10,90 10,60 Z" fill="none" stroke="currentColor" stroke-width="4"/>
<rect x="10" y="25" width="80" height="15" fill="#861211" />
<text x="50" y="36" font-family="Arial" font-weight="bold" font-size="12" fill="white" text-anchor="middle">I M E</text>
<circle cx="50" cy="65" r="18" fill="none" stroke="#2B7574" stroke-width="3" stroke-dasharray="4 2"/>
<polygon points="50,52 54,60 62,60 56,65 58,73 50,68 42,73 44,65 38,60 46,60" fill="currentColor"/>
<circle cx="50" cy="95" r="12" fill="none" stroke="currentColor" stroke-width="2"/>
<path d="M50,83 L50,107 M38,95 L62,95 M44,85 C40,95 40,95 44,105 M56,85 C60,95 60,95 56,105" stroke="currentColor" stroke-width="1.5" fill="none"/>
</svg>"""

st.markdown(f"""
<div class="header-container">
{svg_ime_logo}
<div class="header-title">
<h1>SISTEMA TÁTICO DE SUPORTE À DECISÃO</h1>
<span>Centro de Controle Operacional - CARTIO</span>
</div>
</div>
""", unsafe_allow_html=True)


# --- Funções Auxiliares de Geometria e Física ---
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000
    phi_1, phi_2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def interpolar_rota(G, rota, num_pontos=1000):
    if not rota: return []
    coords = []
    for i in range(len(rota) - 1):
        u, v = rota[i], rota[i+1]
        edge_data = G.get_edge_data(u, v)[0]
        if 'geometry' in edge_data:
            xs, ys = edge_data['geometry'].xy
            coords.extend(list(zip(xs, ys)))
        else:
            coords.extend([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
            
    coords_np = np.array(coords)
    mask = np.ones(len(coords_np), dtype=bool)
    mask[1:] = (coords_np[1:] != coords_np[:-1]).any(axis=1)
    coords_np = coords_np[mask]
    if len(coords_np) < 2: return coords_np.tolist()
        
    distancias = np.zeros(len(coords_np))
    for i in range(1, len(coords_np)):
        distancias[i] = distancias[i-1] + math.hypot(coords_np[i][0] - coords_np[i-1][0], coords_np[i][1] - coords_np[i-1][1])
        
    distancias_interp = np.linspace(0, distancias[-1], num_pontos)
    x_interp = np.interp(distancias_interp, distancias, coords_np[:, 0])
    y_interp = np.interp(distancias_interp, distancias, coords_np[:, 1])
    return list(zip(x_interp, y_interp))


# --- Inicialização ---
@st.cache_resource(show_spinner=False)
def inicializar_grafo():
    G = ox.graph_from_point((-22.898, -43.195), dist=3500, network_type='drive')
    for u, v, key, data in G.edges(keys=True, data=True):
        data['peso_tatico'] = data.get('length', 10.0)
    return G

with st.spinner("Sincronizando malha viária tática..."):
    G_base = inicializar_grafo()

if "G_atual" not in st.session_state: st.session_state.G_atual = G_base.copy()
if "alerta_acionado" not in st.session_state: st.session_state.alerta_acionado = False
if "ponto_bloqueio" not in st.session_state: st.session_state.ponto_bloqueio = None

G = st.session_state.G_atual
nos_grafo = list(G.nodes())

if "origem_padrao" not in st.session_state: st.session_state.origem_padrao = nos_grafo[10] if len(nos_grafo) > 10 else nos_grafo[0]
if "destino_padrao" not in st.session_state: st.session_state.destino_padrao = nos_grafo[-10] if len(nos_grafo) > 10 else nos_grafo[-1]

GATEWAYS_LORAWAN = [
    {"name": "Gateway Centro", "coords": [-43.190, -22.905], "radius": 1500},
    {"name": "Gateway Caju", "coords": [-43.220, -22.880], "radius": 1200}
]

# --- Barra Lateral ---
st.sidebar.title("Comando Logístico")

origem_selecionada = st.sidebar.selectbox("Coordenada de Origem", nos_grafo, index=nos_grafo.index(st.session_state.origem_padrao), disabled=is_demo)
destino_selecionado = st.sidebar.selectbox("Coordenada de Destino", nos_grafo, index=nos_grafo.index(st.session_state.destino_padrao), disabled=is_demo)

st.sidebar.markdown("---")
if not is_demo:
    progresso_missao = st.sidebar.slider("PROGRESSO DA MISSÃO (%)", min_value=0.0, max_value=100.0, value=st.session_state.progresso_missao, step=0.5)
    st.session_state.progresso_missao = progresso_missao
else:
    st.sidebar.slider("PROGRESSO DA MISSÃO (%)", min_value=0.0, max_value=100.0, value=0.0, disabled=True)
st.sidebar.markdown("---")

cenario_alerta = st.sidebar.selectbox("Injeção de Ameaça (LoRaWAN)", [
    "Cenário A: Barricada na Av. Presidente Vargas",
    "Cenário B: Interceptação (Acesso Av. Brasil)",
    "Cenário C: Alagamento em São Cristóvão"
], disabled=is_demo)

def acionar_bloqueio_logico(nome_cenario):
    if "Cenário A" in nome_cenario: coord = (-22.902, -43.190)
    elif "Cenário B" in nome_cenario: coord = (-22.880, -43.218)
    else: coord = (-22.892, -43.220)
    lat, lon = coord
    tem_sinal = any(haversine(lon, lat, gw["coords"][0], gw["coords"][1]) <= gw["radius"] for gw in GATEWAYS_LORAWAN)
    if tem_sinal:
        no_mais_proximo = ox.distance.nearest_nodes(st.session_state.G_atual, X=lon, Y=lat)
        for u, v, key, data in st.session_state.G_atual.edges(no_mais_proximo, keys=True, data=True):
            st.session_state.G_atual[u][v][key]['peso_tatico'] += 999999
        for u, v, key, data in st.session_state.G_atual.in_edges(no_mais_proximo, keys=True, data=True):
            st.session_state.G_atual[u][v][key]['peso_tatico'] += 999999
        st.session_state.alerta_acionado = True
        st.session_state.ponto_bloqueio = (lat, lon)
        return True, (lat, lon)
    else:
        st.session_state.ponto_bloqueio = (lat, lon)
        return False, (lat, lon)

def on_transmitir():
    sucesso, coord = acionar_bloqueio_logico(cenario_alerta)
    if sucesso: st.sidebar.success("ALERTA RECEBIDO NO CCO: Rota reconfigurada com sucesso.")
    else: st.sidebar.error("FALHA DE COMUNICAÇÃO: Fora do alcance dos Gateways.")

st.sidebar.button("TRANSMITIR ALERTA TÁTICO", on_click=on_transmitir, disabled=is_demo, use_container_width=True)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
if st.sidebar.button("▶️ INICIAR DEMO AUTOMATIZADA", disabled=is_demo, type="primary"):
    # Reset state for demo
    st.session_state.demo_mode = True
    st.session_state.G_atual = G_base.copy()
    st.session_state.alerta_acionado = False
    st.session_state.ponto_bloqueio = None
    st.session_state.progresso_missao = 0.0
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8rem;'>"
    "<strong>CARTIO - Simulador Logístico</strong><br>Versão: 5.0<br>Desenvolvido por: <strong>Wagner Calazans</strong>"
    "</div>", unsafe_allow_html=True
)

tab_simulador, tab_arquitetura = st.tabs(["📍 SIMULADOR OPERACIONAL", "⚙️ ARQUITETURA DE BORDA (EDGE AI)"])

with tab_simulador:
    # Containers para a Demo
    narrative_box = st.empty()
    metrics_box = st.empty()
    map_box = st.empty()

    def build_deck(G_current, progresso, cenario_block=None, z=14, p=50, cam_lat=-22.898, cam_lon=-43.195, escurecer=False):
        try:
            rota_segura = nx.shortest_path(G_current, source=origem_selecionada, target=destino_selecionado, weight='peso_tatico')
            dist_m = sum([G_current[rota_segura[i]][rota_segura[i+1]][0].get('length', 0) for i in range(len(rota_segura)-1)])
            tmp_min = (dist_m / 1000) / 40 * 60
            r_interp = interpolar_rota(G_current, rota_segura, 1000)
        except nx.NetworkXNoPath:
            r_interp, dist_m, tmp_min = [], 0, 0

        posicoes = []
        tem_sinal = False
        if r_interp:
            idx = int((progresso / 100.0) * (len(r_interp) - 1))
            for i in range(4):
                posicoes.append(r_interp[max(0, idx - (i * 15))])
            if posicoes:
                tem_sinal = any(haversine(posicoes[0][0], posicoes[0][1], gw["coords"][0], gw["coords"][1]) <= gw["radius"] for gw in GATEWAYS_LORAWAN)

        layers = []
        # Gateways
        cor_gw = [18, 72, 76, 20] if escurecer else [18, 72, 76, 50]
        layers.append(pdk.Layer("ScatterplotLayer", data=[{"coords": g["coords"], "radius": g["radius"]} for g in GATEWAYS_LORAWAN],
                                get_position="coords", get_radius="radius", get_fill_color=cor_gw, get_line_color=[43, 117, 116, 100], lineWidthMinPixels=1))
        # Rota
        if r_interp:
            cor_rota = [105, 105, 105, 150] if escurecer else ([43, 117, 116, 255] if st.session_state.alerta_acionado else [226, 226, 224, 200])
            layers.append(pdk.Layer("PathLayer", data=[{"path": r_interp}], get_path="path", get_color=cor_rota, width_scale=20, width_min_pixels=4, get_width=5))
        
        # Comboio
        if posicoes:
            cor_comb = [105, 105, 105, 150] if escurecer else [226, 226, 224, 255]
            layers.append(pdk.Layer("ScatterplotLayer", data=[{"coords": p} for p in posicoes], get_position="coords", get_radius=30, get_fill_color=cor_comb, get_line_color=[14, 41, 49, 255], radius_min_pixels=6, lineWidthMinPixels=2))
        
        # Bloqueio
        if cenario_block:
            layers.append(pdk.Layer("ScatterplotLayer", data=[{"coords": [cenario_block[1], cenario_block[0]]}], get_position="coords", get_radius=60, get_fill_color=[134, 18, 17, 220], radius_min_pixels=8, stroked=True, lineWidthMinPixels=2, get_line_color=[255, 255, 255, 255]))

        view_state = pdk.ViewState(latitude=cam_lat, longitude=cam_lon, zoom=z, pitch=p, bearing=0)
        return pdk.Deck(layers=layers, initial_view_state=view_state, map_style=pdk.map_styles.DARK), tem_sinal, dist_m, tmp_min

    def render_metrics(container, tem_sinal, dist, tmp, obs):
        col1, col2, col3, col4 = container.columns(4)
        status_html = "<span class='status-online'>SISTEMA ONLINE</span>" if tem_sinal else "<span class='status-offline'>SINAL DEGRADADO</span>"
        col1.markdown(f"**STATUS LORAWAN:**<br><br>{status_html}", unsafe_allow_html=True)
        col2.metric("DISTÂNCIA DA ROTA (KM)", f"{dist / 1000:.2f}")
        col3.metric("TEMPO ESTIMADO (MIN)", f"{tmp:.1f}")
        col4.metric("OBSTÁCULOS DETECTADOS", f"{obs}")

    if st.session_state.demo_mode:
        # ---- FILME INTERATIVO (DEMO) ----
        # CENA 1
        narrative_box.info("🎬 **CENA 1:** Rio de Janeiro, Sexta-feira, 5 de junho de 2026, 18h23min. Chuva forte de verão, alagamentos em algumas regiões da cidade.")
        deck, ts, d, t = build_deck(st.session_state.G_atual, 0.0, None, z=11, p=0, cam_lat=-22.898, cam_lon=-43.195)
        render_metrics(metrics_box, ts, d, t, 0)
        map_box.pydeck_chart(deck)
        time.sleep(5)

        # CENA 2
        narrative_box.warning("🎬 **CENA 2:** Rio de Janeiro, Sexta-feira, 18h32min. O sistema elétrico cai. Sinal de celular e internet 4G/5G ficam **sem operação**.")
        deck, ts, d, t = build_deck(st.session_state.G_atual, 0.0, None, z=12, p=0, cam_lat=-22.898, cam_lon=-43.195, escurecer=True)
        render_metrics(metrics_box, False, d, t, 0) # Força offline na narrativa
        map_box.pydeck_chart(deck)
        time.sleep(5)

        # CENA 3
        local_txt = cenario_alerta.split(':')[1].strip()
        narrative_box.error(f"🎬 **CENA 3:** {local_txt}, Sexta-feira, 19h02min. Viatura tática se aproxima e identifica bloqueio via **YOLOv5** na borda! Transmitindo pacote de emergência via **LoRaWAN** para o CCO.")
        sucesso, coord_bloqueio = acionar_bloqueio_logico(cenario_alerta)
        
        # Piscar o radar Lorawan
        for _ in range(3):
            deck, ts, d, t = build_deck(st.session_state.G_atual, 0.0, coord_bloqueio, z=14.5, p=60, cam_lat=coord_bloqueio[0], cam_lon=coord_bloqueio[1])
            render_metrics(metrics_box, True, d, t, 1)
            map_box.pydeck_chart(deck)
            time.sleep(0.5)

        time.sleep(2)

        # CENA 4
        narrative_box.success("🎬 **CENA 4:** Alerta validado pelo Gateway CARTIO! Rota recalculada com sucesso via NetworkX. Desviando comboio pelo caminho mais rápido e seguro.")
        for prg in range(0, 101, 2): # Move de 2 em 2%
            # Câmera acompanha o comboio
            deck, ts, d, t = build_deck(st.session_state.G_atual, prg, coord_bloqueio, z=15, p=50, cam_lat=-22.898, cam_lon=-43.195) 
            
            # Recupera as posições temporárias só para guiar a câmera
            r_interp = interpolar_rota(st.session_state.G_atual, nx.shortest_path(st.session_state.G_atual, origem_selecionada, destino_selecionado, weight='peso_tatico'), 1000)
            if r_interp:
                idx = int((prg / 100.0) * (len(r_interp) - 1))
                c_lon, c_lat = r_interp[idx]
                deck.initial_view_state.latitude = c_lat
                deck.initial_view_state.longitude = c_lon

            render_metrics(metrics_box, ts, d, t, 1)
            map_box.pydeck_chart(deck)
            time.sleep(0.15)
            
        narrative_box.info("🎬 **FIM DA SIMULAÇÃO.** O comboio de suprimentos atingiu o destino com sucesso!")
        time.sleep(3)
        st.session_state.demo_mode = False
        st.rerun()

    else:
        # ---- MODO INTERATIVO NORMAL ----
        # Se bloqueio ativo (botão manual)
        coord_blq = st.session_state.ponto_bloqueio if st.session_state.alerta_acionado else None
        
        # Recuperar a camera acompanhando o comboio manual
        r_interp = interpolar_rota(st.session_state.G_atual, nx.shortest_path(st.session_state.G_atual, origem_selecionada, destino_selecionado, weight='peso_tatico') if st.session_state.G_atual.has_node(origem_selecionada) and st.session_state.G_atual.has_node(destino_selecionado) else [], 1000)
        c_lon, c_lat = (-43.195, -22.898)
        if r_interp:
            idx = int((st.session_state.progresso_missao / 100.0) * (len(r_interp) - 1))
            c_lon, c_lat = r_interp[idx]

        deck, ts, d, t = build_deck(st.session_state.G_atual, st.session_state.progresso_missao, coord_blq, z=14, p=50, cam_lat=c_lat, cam_lon=c_lon)
        render_metrics(metrics_box, ts, d, t, 1 if st.session_state.alerta_acionado else 0)
        map_box.pydeck_chart(deck)


with tab_arquitetura:
    st.markdown("### Integração de Visão Computacional de Borda e Roteamento Dinâmico")
    st.info("""
    **O Escopo da Simulação (CARTIO):**
    O processo tem início com a captura de imagens pela câmera veicular e o processamento local (Edge) pelo modelo **YOLOv5**, que extrai as coordenadas da anomalia na pista. Em seguida, este pacote georreferenciado é transmitido via Rádio LoRaWAN para o Centro de Controle (CCO). Por fim, o servidor recebe a notificação, recalcula as rotas de emergência utilizando algoritmos em grafos (NetworkX) e plota a atualização tática no mapa interativo.
    """)
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### Diagrama Lógico da Solução")
        codigo_mermaid = """
        ```mermaid
        graph TD
            classDef edge fill:#0E2931,stroke:#2B7574,stroke-width:2px,color:#E2E2E0
            classDef lora fill:#861211,stroke:#E2E2E0,stroke-width:2px,color:#E2E2E0
            classDef cco fill:#12484C,stroke:#2B7574,stroke-width:2px,color:#E2E2E0
            classDef base fill:transparent,stroke:#861211,stroke-width:1px,stroke-dasharray: 5 5
            
            subgraph Viatura Tática (Borda)
                direction TB
                CAM[Camera Raspberry v3] -->|Captura de Frames| RPI
                RPI[Raspberry Pi 5 8GB/16GB\nYOLOv5 Nano + GPS]:::edge
                
                subgraph Ambiente Alpine Linux 64x
                    direction TB
                    A1[python3, py3-pip]
                    A2[libonnxruntime]
                    A3[opencv / pyserial]
                end
                
                RPI -.-> Ambiente Alpine Linux 64x
            end
            
            subgraph Transmissão RF
                direction TB
                LORA[Rádio LoRaWAN\nRYLR998 Reyax\n915MHz - OTAA]:::lora
            end
            
            RPI -->|Coordenada Georreferenciada| LORA
            LORA == Transmissão CARTIO ==> CCO
            
            subgraph Servidor CARTIO
                direction TB
                CCO[Centro de Controle\nStreamlit + NetworkX\nRecálculo Dinâmico]:::cco
            end
        ```
        """
        st.markdown(codigo_mermaid)
    with col2:
        st.markdown("#### Trabalhos Futuros e Evolução")
        st.markdown("""
        O simulador CARTIO serve como fundação de testes para melhorias contínuas:
        - **Autocorrelação Espacial (PySAL / LISA):** Automatizar a identificação de *hotspots* regionais no GeoPackage.
        - **Função de Custo Expandida (OSMnx / NetworkX):** Implementar algoritmo que penaliza o peso tático das vias vizinhas.
        - **Validação de Redes (CARTIO / LoRaWAN):** Avaliar o impacto direto do atraso de rede (delay) na plotagem de rotas criptografadas.
        """)
