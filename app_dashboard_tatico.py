"""
============================================================
Projeto CARTIO
Autoria: Wagner Calazans
Ano de criação: 2026
Versao: 1.0 (Dashboard Tático de Roteamento)
IME - Instituto Militar de Engenharia
Arquivo: app_dashboard_tatico.py
Descrição: Sistema Tático de Suporte à Decisão Logística para
roteamento dinâmico sob redes degradadas.
============================================================
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
import json

# --- Configurações da Interface Web ---
st.set_page_config(
    page_title="Dashboard Tático - CARTIO", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Funções de Inicialização e Lógica de Negócio ---

@st.cache_resource(show_spinner=False)
def inicializar_grafo():
    """
    Inicialização: Carrega um grafo viário de exemplo utilizando a biblioteca osmnx.
    Utilizando um recorte urbano pequeno para garantir performance.
    """
    # Ponto focal: Saída do Centro do Rio em direção à Baixada (Central do Brasil / Av. Brasil)
    point = (-22.898, -43.195)
    
    # Carrega a malha viária num raio de 3500m (pegando Centro, São Cristóvão, Caju - acesso à Baixada)
    # Nota: Distâncias muito maiores (ex: 30km até Caxias) excedem o limite de 1GB de RAM do Streamlit gratuito
    G = ox.graph_from_point(point, dist=3500, network_type='drive')
    
    # Inicializa a variável 'peso_tatico' baseada no comprimento ('length')
    for u, v, key, data in G.edges(keys=True, data=True):
        data['peso_tatico'] = data.get('length', 10.0)
        
    return G

# Carrega o grafo base
with st.spinner("Carregando malha viária tática..."):
    G_base = inicializar_grafo()

# Utiliza session_state para gerenciar as alterações dinâmicas na malha
if "G_atual" not in st.session_state:
    st.session_state.G_atual = G_base.copy()

if "alerta_acionado" not in st.session_state:
    st.session_state.alerta_acionado = False
if "ponto_bloqueio" not in st.session_state:
    st.session_state.ponto_bloqueio = None

G = st.session_state.G_atual

# Pega a lista de nós do grafo para os seletores
nos_grafo = list(G.nodes())

# Define origem e destino padrão para visualização inicial
if "origem_padrao" not in st.session_state:
    st.session_state.origem_padrao = nos_grafo[10] if len(nos_grafo) > 10 else nos_grafo[0]
if "destino_padrao" not in st.session_state:
    st.session_state.destino_padrao = nos_grafo[-10] if len(nos_grafo) > 10 else nos_grafo[-1]


# --- Componentes da Interface de Usuário (UI) ---

# 1. Barra Lateral (Sidebar)
st.sidebar.title("Comando e Controle")
st.sidebar.subheader("Rotas Logísticas")

origem_selecionada = st.sidebar.selectbox(
    "Ponto de Origem", 
    nos_grafo, 
    index=nos_grafo.index(st.session_state.origem_padrao)
)

destino_selecionado = st.sidebar.selectbox(
    "Ponto de Destino", 
    nos_grafo, 
    index=nos_grafo.index(st.session_state.destino_padrao)
)

cenario_alerta = st.sidebar.selectbox(
    "Cenário de Ameaça (LoRaWAN)",
    [
        "Cenário 1: Barricada na Av. Presidente Vargas",
        "Cenário 2: Veículo Interceptado no Caju (Acesso Av. Brasil)",
        "Cenário 3: Alagamento em São Cristóvão"
    ]
)

filtro_severidade = st.sidebar.selectbox(
    "Filtro de Severidade do Alerta", 
    ["Alta", "Média", "Baixa"]
)

def simular_recebimento_alerta():
    """
    Simulação do Alerta: Função fictícia que simula a chegada de um JSON via rádio LoRaWAN.
    """
    # Define a coordenada baseada no cenário escolhido
    if "Cenário 1" in cenario_alerta:
        coord_simulada = (-22.902, -43.190) # Pres. Vargas
    elif "Cenário 2" in cenario_alerta:
        coord_simulada = (-22.880, -43.218) # Caju / Av Brasil
    else:
        coord_simulada = (-22.892, -43.220) # São Cristóvão

    # JSON fictício simulando recepção via rádio
    json_recebido = {
        "id_dispositivo": "LORA-NODE-01",
        "timestamp": "2026-06-07T12:00:00Z",
        "coordenada": coord_simulada,
        "severidade": filtro_severidade,
        "token_seguranca": "CARTIO"
    }

    # Verificação condicional validando o Token CARTIO
    if json_recebido.get("token_seguranca") == "CARTIO":
        
        # 1. Converter a coordenada recebida em um ponto no grafo
        lat, lon = json_recebido["coordenada"]
        no_mais_proximo = ox.distance.nearest_nodes(st.session_state.G_atual, X=lon, Y=lat)
        
        # 2. Aumentar drasticamente a variável "peso_tatico" (ou length) daquela aresta específica
        # Aumentamos o peso tático das arestas que conectam ao nó bloqueado
        for u, v, key, data in st.session_state.G_atual.edges(no_mais_proximo, keys=True, data=True):
            st.session_state.G_atual[u][v][key]['peso_tatico'] += 999999
        for u, v, key, data in st.session_state.G_atual.in_edges(no_mais_proximo, keys=True, data=True):
            st.session_state.G_atual[u][v][key]['peso_tatico'] += 999999

        # Atualiza a interface
        st.session_state.alerta_acionado = True
        st.session_state.ponto_bloqueio = (lat, lon)
        
    else:
        st.sidebar.error("Alerta descartado: Falha na validação do Token de Segurança.")

st.sidebar.button("Simular Recebimento de Alerta LoRaWAN", on_click=simular_recebimento_alerta)


# --- Leitura de Dados (Camada de Alertas) ---
# O sistema prevê a leitura de um arquivo 'grafo_tatico.gpkg' utilizando geopandas
try:
    gdf_alertas = gpd.read_file("grafo_tatico.gpkg", layer="alertas")
    alertas_geopackage_status = "Carregado com sucesso"
except Exception as e:
    alertas_geopackage_status = "Arquivo 'grafo_tatico.gpkg' não encontrado. Rodando simulação interna."

st.sidebar.markdown(f"<small>Status GeoPackage: {alertas_geopackage_status}</small>", unsafe_allow_html=True)

# --- Recálculo Dinâmico (Back-end) ---
try:
    # Roda o algoritmo de caminho mínimo imediatamente entre Origem e Destino
    rota_segura = nx.shortest_path(
        st.session_state.G_atual, 
        source=origem_selecionada, 
        target=destino_selecionado, 
        weight='peso_tatico'
    )
    
    # Calcula distância da rota
    distancia_rota_m = 0
    for i in range(len(rota_segura) - 1):
        u, v = rota_segura[i], rota_segura[i+1]
        distancia_rota_m += st.session_state.G_atual[u][v][0].get('length', 0)
        
    tempo_estimado_min = (distancia_rota_m / 1000) / 40 * 60 # ETA a 40 km/h médio

except nx.NetworkXNoPath:
    rota_segura = []
    distancia_rota_m = 0
    tempo_estimado_min = 0

# 2. Painel de Métricas (Top Metrics)
st.markdown("### Sistema Tático de Suporte à Decisão Logística")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Distância Total da Rota (km)", f"{distancia_rota_m / 1000:.2f} km")
with col2:
    st.metric("Tempo Estimado de Viagem (ETA)", f"{tempo_estimado_min:.1f} min")
with col3:
    # Se o alerta for acionado, consideramos que 1 obstáculo foi evitado na nova rota
    num_obstaculos_evitados = 1 if st.session_state.alerta_acionado else 0
    st.metric("Obstáculos Evitados", f"{num_obstaculos_evitados}")


# 3. Mapa Central (Atualização Visual)
if len(rota_segura) > 0:
    origem_no = G.nodes[origem_selecionada]
    centro_mapa = [origem_no['y'], origem_no['x']]
else:
    centro_mapa = [-22.898, -43.195]

m = folium.Map(location=centro_mapa, zoom_start=15, tiles="CartoDB dark_matter")

if len(rota_segura) > 0:
    coords_rota = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in rota_segura]
    
    # A rota original desaparece sendo substituída pela nova rota segura (linha azul espessa)
    if st.session_state.alerta_acionado:
        cor_linha = "#00BFFF" # Azul espesso (Deep Sky Blue) para rota segura
        espessura_linha = 8
    else:
        cor_linha = "#808080" # Cinza para rota original
        espessura_linha = 5
        
    folium.PolyLine(
        coords_rota,
        color=cor_linha,
        weight=espessura_linha,
        opacity=0.9,
        tooltip="Rota de Comboio Logístico"
    ).add_to(m)
    
    # Marcadores de Início e Fim
    folium.Marker(coords_rota[0], tooltip="Ponto de Origem", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(coords_rota[-1], tooltip="Ponto de Destino", icon=folium.Icon(color="darkblue", icon="flag")).add_to(m)

# O ponto do bloqueio recebido via rádio deve ser plotado como um marcador vermelho com um ícone de alerta.
if st.session_state.alerta_acionado and st.session_state.ponto_bloqueio is not None:
    folium.Marker(
        st.session_state.ponto_bloqueio,
        tooltip="Bloqueio Recebido via Rádio",
        icon=folium.Icon(color="red", icon="warning", prefix="fa")
    ).add_to(m)

# Ocupe o restante da tela com o mapa interativo
st_folium(m, width="100%", height=600, returned_objects=[])
