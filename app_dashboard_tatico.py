"""
============================================================
Projeto CARTIO
Autoria: Wagner Calazans
Ano de criação: 2026
Versao: 2.0 (Simulador Tático Multi-Agente)
IME - Instituto Militar de Engenharia
Arquivo: app_dashboard_tatico.py
Descrição: Sistema Tático de Suporte à Decisão Logística para
roteamento dinâmico sob redes degradadas e física LoRaWAN.
============================================================
"""

import streamlit as st
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pydeck as pdk
import json
import math
import numpy as np

# --- Configurações da Interface Web ---
st.set_page_config(
    page_title="Simulador Tático - CARTIO", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Funções Auxiliares de Geometria e Física ---
def haversine(lon1, lat1, lon2, lat2):
    """Calcula a distância (em metros) entre dois pontos na Terra."""
    R = 6371000 # Raio da Terra em metros
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def interpolar_rota(G, rota, num_pontos=1000):
    """Gera pontos interpolados de alta resolução ao longo da rota."""
    if not rota:
        return []
    coords = []
    for i in range(len(rota) - 1):
        u = rota[i]
        v = rota[i+1]
        
        # Pega a geometria real da aresta, se existir, senão linha reta entre os nós
        edge_data = G.get_edge_data(u, v)[0]
        if 'geometry' in edge_data:
            xs, ys = edge_data['geometry'].xy
            coords.extend(list(zip(xs, ys)))
        else:
            p1 = (G.nodes[u]['x'], G.nodes[u]['y'])
            p2 = (G.nodes[v]['x'], G.nodes[v]['y'])
            coords.append(p1)
            coords.append(p2)
            
    # Simplifica e interpola linearmente para num_pontos (aprox)
    if not coords:
        return []
        
    # Extrai array numpy para interpolação
    coords_np = np.array(coords)
    # Remove duplicatas sequenciais
    mask = np.ones(len(coords_np), dtype=bool)
    mask[1:] = (coords_np[1:] != coords_np[:-1]).any(axis=1)
    coords_np = coords_np[mask]
    
    if len(coords_np) < 2:
        return coords_np.tolist()
        
    # Calcula distâncias cumulativas
    distancias = np.zeros(len(coords_np))
    for i in range(1, len(coords_np)):
        distancias[i] = distancias[i-1] + math.hypot(coords_np[i][0] - coords_np[i-1][0], coords_np[i][1] - coords_np[i-1][1])
        
    distancia_total = distancias[-1]
    
    # Gera novos pontos uniformemente espaçados
    distancias_interp = np.linspace(0, distancia_total, num_pontos)
    x_interp = np.interp(distancias_interp, distancias, coords_np[:, 0])
    y_interp = np.interp(distancias_interp, distancias, coords_np[:, 1])
    
    return list(zip(x_interp, y_interp))


# --- Inicialização e Lógica de Negócio ---

@st.cache_resource(show_spinner=False)
def inicializar_grafo():
    """Carrega o grafo viário base focado no Centro -> Baixada."""
    point = (-22.898, -43.195) # Central do Brasil
    G = ox.graph_from_point(point, dist=3500, network_type='drive')
    for u, v, key, data in G.edges(keys=True, data=True):
        data['peso_tatico'] = data.get('length', 10.0)
    return G

with st.spinner("Carregando malha viária tática (Simulador)..."):
    G_base = inicializar_grafo()

if "G_atual" not in st.session_state:
    st.session_state.G_atual = G_base.copy()
if "alerta_acionado" not in st.session_state:
    st.session_state.alerta_acionado = False
if "ponto_bloqueio" not in st.session_state:
    st.session_state.ponto_bloqueio = None

G = st.session_state.G_atual
nos_grafo = list(G.nodes())

if "origem_padrao" not in st.session_state:
    st.session_state.origem_padrao = nos_grafo[10] if len(nos_grafo) > 10 else nos_grafo[0]
if "destino_padrao" not in st.session_state:
    st.session_state.destino_padrao = nos_grafo[-10] if len(nos_grafo) > 10 else nos_grafo[-1]


# --- Definição dos Gateways LoRaWAN (Física do Sinal) ---
GATEWAYS_LORAWAN = [
    {"name": "Gateway Centro", "coords": [-43.190, -22.905], "radius": 1500},
    {"name": "Gateway Caju", "coords": [-43.220, -22.880], "radius": 1200}
]


# --- 1. Barra Lateral (Sidebar) ---
st.sidebar.title("Comando e Controle")
st.sidebar.markdown("---")

origem_selecionada = st.sidebar.selectbox("Ponto de Origem", nos_grafo, index=nos_grafo.index(st.session_state.origem_padrao))
destino_selecionado = st.sidebar.selectbox("Ponto de Destino", nos_grafo, index=nos_grafo.index(st.session_state.destino_padrao))

st.sidebar.markdown("---")
st.sidebar.subheader("Simulador do Comboio")

# Controle de Tempo/Progresso da Missão
progresso_missao = st.sidebar.slider("Progresso da Missão (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)

cenario_alerta = st.sidebar.selectbox(
    "Injetar Ameaça no Terreno",
    [
        "Cenário 1: Barricada na Av. Presidente Vargas",
        "Cenário 2: Caminhão Atravessado (Acesso Av. Brasil)",
        "Cenário 3: Alagamento em São Cristóvão"
    ]
)

def simular_recebimento_alerta():
    if "Cenário 1" in cenario_alerta:
        coord_simulada = (-22.902, -43.190)
    elif "Cenário 2" in cenario_alerta:
        coord_simulada = (-22.880, -43.218)
    else:
        coord_simulada = (-22.892, -43.220)

    lat, lon = coord_simulada
    
    # 1. Verifica Física de Sinal: O bloqueio ocorreu dentro da área de um Gateway?
    tem_sinal = False
    for gw in GATEWAYS_LORAWAN:
        dist = haversine(lon, lat, gw["coords"][0], gw["coords"][1])
        if dist <= gw["radius"]:
            tem_sinal = True
            break
            
    if tem_sinal:
        # Ponto com cobertura: O rádio envia a mensagem e o CCO recalcula
        no_mais_proximo = ox.distance.nearest_nodes(st.session_state.G_atual, X=lon, Y=lat)
        for u, v, key, data in st.session_state.G_atual.edges(no_mais_proximo, keys=True, data=True):
            st.session_state.G_atual[u][v][key]['peso_tatico'] += 999999
        for u, v, key, data in st.session_state.G_atual.in_edges(no_mais_proximo, keys=True, data=True):
            st.session_state.G_atual[u][v][key]['peso_tatico'] += 999999

        st.session_state.alerta_acionado = True
        st.session_state.ponto_bloqueio = (lat, lon)
        st.sidebar.success("Alerta Transmitido via LoRaWAN! Rota recalculada.")
    else:
        st.session_state.ponto_bloqueio = (lat, lon)
        st.sidebar.error("ZONA CEGA: Comboio fora da cobertura LoRaWAN. Alerta não recebido pelo CCO.")

st.sidebar.button("Injetar Ameaça e Transmitir (LoRaWAN)", on_click=simular_recebimento_alerta)


# --- Back-end de Roteamento ---
try:
    rota_segura = nx.shortest_path(st.session_state.G_atual, source=origem_selecionada, target=destino_selecionado, weight='peso_tatico')
    distancia_rota_m = sum([st.session_state.G_atual[rota_segura[i]][rota_segura[i+1]][0].get('length', 0) for i in range(len(rota_segura)-1)])
    tempo_estimado_min = (distancia_rota_m / 1000) / 40 * 60
    
    # Gera a linha interpolada (1000 pontos)
    rota_interpolada = interpolar_rota(G, rota_segura, num_pontos=1000)
except nx.NetworkXNoPath:
    rota_segura = []
    rota_interpolada = []
    distancia_rota_m = 0
    tempo_estimado_min = 0

# Calcula a posição atual do comboio baseada no progresso
posicoes_comboio = []
tem_sinal_agora = False

if rota_interpolada:
    idx_lider = int((progresso_missao / 100.0) * (len(rota_interpolada) - 1))
    
    # Frota de 4 veículos: Líder e 3 veículos de retaguarda (espaçados na rota)
    espacamento_tatico = 15 # pontos de distância entre os carros
    for i in range(4):
        idx_carro = max(0, idx_lider - (i * espacamento_tatico))
        posicoes_comboio.append(rota_interpolada[idx_carro])
        
    # Verifica Cobertura do Carro Líder
    lon_lider, lat_lider = posicoes_comboio[0]
    for gw in GATEWAYS_LORAWAN:
        dist = haversine(lon_lider, lat_lider, gw["coords"][0], gw["coords"][1])
        if dist <= gw["radius"]:
            tem_sinal_agora = True
            break

# --- 2. Painel de Métricas (Top Metrics) ---
st.markdown("### Simulador de Roteamento Multi-Veículos (PyDeck)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Sinal LoRaWAN", "🟢 ONLINE" if tem_sinal_agora else "🔴 OFFLINE")
with col2:
    st.metric("Distância (km)", f"{distancia_rota_m / 1000:.2f}")
with col3:
    st.metric("Tempo Est. (ETA)", f"{tempo_estimado_min:.1f} min")
with col4:
    num_obstaculos = 1 if st.session_state.alerta_acionado else 0
    st.metric("Obstáculos Desviados", f"{num_obstaculos}")


# --- 3. Renderização Cinematográfica (PyDeck) ---
layers = []

# Camada 1: Círculos de Cobertura LoRaWAN (Física do Sinal)
gw_data = [{"coords": gw["coords"], "radius": gw["radius"]} for gw in GATEWAYS_LORAWAN]
layer_gateways = pdk.Layer(
    "ScatterplotLayer",
    data=gw_data,
    get_position="coords",
    get_radius="radius",
    get_fill_color=[0, 255, 127, 30], # Verde Translúcido
    get_line_color=[0, 255, 127, 100],
    lineWidthMinPixels=2,
    pickable=True
)
layers.append(layer_gateways)

# Camada 2: Rota Traçada (Glow)
if rota_interpolada:
    layer_rota = pdk.Layer(
        "PathLayer",
        data=[{"path": rota_interpolada}],
        get_path="path",
        get_color=[0, 191, 255, 200] if st.session_state.alerta_acionado else [105, 105, 105, 200], # Azul ciano ou Cinza
        width_scale=20,
        width_min_pixels=4,
        get_width=5
    )
    layers.append(layer_rota)

# Camada 3: Comboio de Veículos (Marcadores Vermelhos pulsantes / Amarelos)
if posicoes_comboio:
    comboio_data = [{"coords": pos} for pos in posicoes_comboio]
    layer_comboio = pdk.Layer(
        "ScatterplotLayer",
        data=comboio_data,
        get_position="coords",
        get_radius=30,
        get_fill_color=[255, 140, 0, 255], # Laranja (Veículos Táticos)
        get_line_color=[255, 255, 255, 255],
        radius_min_pixels=6,
        radius_max_pixels=15,
        lineWidthMinPixels=1
    )
    layers.append(layer_comboio)

# Camada 4: Ameaça / Bloqueio
if st.session_state.ponto_bloqueio is not None:
    blq_lat, blq_lon = st.session_state.ponto_bloqueio
    layer_bloqueio = pdk.Layer(
        "ScatterplotLayer",
        data=[{"coords": [blq_lon, blq_lat]}],
        get_position="coords",
        get_radius=60,
        get_fill_color=[255, 0, 0, 200], # Vermelho Vivo
        radius_min_pixels=8,
        stroked=True,
        lineWidthMinPixels=2,
        get_line_color=[255, 255, 255]
    )
    layers.append(layer_bloqueio)

# Define a visualização da Câmera em 3D
if rota_interpolada:
    # A câmera segue o veículo líder
    cam_lon, cam_lat = posicoes_comboio[0]
else:
    cam_lon, cam_lat = -43.195, -22.898

view_state = pdk.ViewState(
    latitude=cam_lat,
    longitude=cam_lon,
    zoom=14,
    pitch=50,      # Ângulo inclinado para efeito 3D e cinematográfico
    bearing=0      # Orientação da câmera
)

r = pdk.Deck(
    layers=layers,
    initial_view_state=view_state,
    map_style="mapbox://styles/mapbox/dark-v11",
    tooltip={"text": "Coordenada: {coords}"}
)

# Renderiza o Deck
st.pydeck_chart(r)

st.markdown("---")
st.markdown("<small>**Nota sobre Física LoRaWAN**: Mova o *Slider* de Progresso na barra lateral. O Status do Sinal altera dinamicamente com base na proximidade física do veículo líder em relação aos Gateways LoRaWAN (círculos verdes). Injetar ameaças fora da zona verde resultará em perda de pacote de alerta no CCO.</small>", unsafe_allow_html=True)
