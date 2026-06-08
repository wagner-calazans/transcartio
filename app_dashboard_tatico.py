"""
============================================================
Projeto CARTIO
Autoria: Wagner Calazans
Ano de criação: 2026
Versao: 4.1 (Tema Dinâmico e Arquitetura Edge Computing)
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

# --- Configurações da Interface Web ---
st.set_page_config(
    page_title="Simulador Tático - CARTIO", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- Estilização CSS: Tema Dinâmico Responsivo ---
# Removidos os fundos fixos (Preto Absoluto) para respeitar o tema Claro/Escuro do SO do usuário.
estilo_css = """
<style>
    /* Cabeçalho Flexível (Transparente para se adaptar ao Claro/Escuro) */
    .header-container {
        display: flex;
        align-items: center;
        padding: 1rem 1rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border-bottom: 3px solid #861211; /* Accento Grená */
        background-color: rgba(43, 117, 116, 0.1); /* Fundo Teal bem leve e translúcido */
    }
    .header-title {
        display: flex;
        flex-direction: column;
    }
    .header-title h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .header-title span {
        font-size: 0.9rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #2B7574; /* Azul Petróleo Claro */
    }
    
    /* Badges de Status Neutras */
    .status-online {
        background-color: #2B7574; /* Teal */
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .status-offline {
        background-color: #861211; /* Grená */
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.9rem;
        font-weight: bold;
    }
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

html_header = f"""
<div class="header-container">
{svg_ime_logo}
<div class="header-title">
<h1>SISTEMA TÁTICO DE SUPORTE À DECISÃO</h1>
<span>Centro de Controle Operacional - CARTIO</span>
</div>
</div>
"""
st.markdown(html_header, unsafe_allow_html=True)

# --- Menu Principal por Abas ---
tab_simulador, tab_arquitetura = st.tabs([
    "📍 SIMULADOR OPERACIONAL", 
    "⚙️ ARQUITETURA DE BORDA (EDGE AI)"
])

# ==========================================
# ABA 1: SIMULADOR TÁTICO
# ==========================================
with tab_simulador:

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
        point = (-22.898, -43.195)
        G = ox.graph_from_point(point, dist=3500, network_type='drive')
        for u, v, key, data in G.edges(keys=True, data=True):
            data['peso_tatico'] = data.get('length', 10.0)
        return G

    with st.spinner("Sincronizando malha viária tática..."):
        G_base = inicializar_grafo()

    if "G_atual" not in st.session_state:
        st.session_state.G_atual = G_base.copy()
    if "alerta_acionado" not in st.session_state:
        st.session_state.alerta_acionado = False
    if "ponto_bloqueio" not in st.session_state:
        st.session_state.ponto_bloqueio = None

    G = st.session_state.G_atual
    nos_grafo = list(G.nodes())

    if "origem_padrao" not in st.session_state: st.session_state.origem_padrao = nos_grafo[10] if len(nos_grafo) > 10 else nos_grafo[0]
    if "destino_padrao" not in st.session_state: st.session_state.destino_padrao = nos_grafo[-10] if len(nos_grafo) > 10 else nos_grafo[-1]

    GATEWAYS_LORAWAN = [
        {"name": "Gateway Centro", "coords": [-43.190, -22.905], "radius": 1500},
        {"name": "Gateway Caju", "coords": [-43.220, -22.880], "radius": 1200}
    ]

    # --- Barra Lateral (Comando Logístico) ---
    st.sidebar.title("Comando Logístico")

    origem_selecionada = st.sidebar.selectbox("Coordenada de Origem", nos_grafo, index=nos_grafo.index(st.session_state.origem_padrao))
    destino_selecionado = st.sidebar.selectbox("Coordenada de Destino", nos_grafo, index=nos_grafo.index(st.session_state.destino_padrao))

    st.sidebar.markdown("---")
    progresso_missao = st.sidebar.slider("PROGRESSO DA MISSÃO (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5)
    st.sidebar.markdown("---")

    cenario_alerta = st.sidebar.selectbox("Injeção de Ameaça (LoRaWAN)", [
        "Cenário A: Barricada na Av. Presidente Vargas",
        "Cenário B: Interceptação (Acesso Av. Brasil)",
        "Cenário C: Alagamento em São Cristóvão"
    ])

    def simular_recebimento_alerta():
        if "Cenário A" in cenario_alerta: coord_simulada = (-22.902, -43.190)
        elif "Cenário B" in cenario_alerta: coord_simulada = (-22.880, -43.218)
        else: coord_simulada = (-22.892, -43.220)

        lat, lon = coord_simulada
        tem_sinal = any(haversine(lon, lat, gw["coords"][0], gw["coords"][1]) <= gw["radius"] for gw in GATEWAYS_LORAWAN)
                
        if tem_sinal:
            no_mais_proximo = ox.distance.nearest_nodes(st.session_state.G_atual, X=lon, Y=lat)
            for u, v, key, data in st.session_state.G_atual.edges(no_mais_proximo, keys=True, data=True):
                st.session_state.G_atual[u][v][key]['peso_tatico'] += 999999
            for u, v, key, data in st.session_state.G_atual.in_edges(no_mais_proximo, keys=True, data=True):
                st.session_state.G_atual[u][v][key]['peso_tatico'] += 999999

            st.session_state.alerta_acionado = True
            st.session_state.ponto_bloqueio = (lat, lon)
            st.sidebar.success("ALERTA RECEBIDO NO CCO: Rota reconfigurada com sucesso.")
        else:
            st.session_state.ponto_bloqueio = (lat, lon)
            st.sidebar.error("FALHA DE COMUNICAÇÃO: Fora do alcance dos Gateways.")

    st.sidebar.button("TRANSMITIR ALERTA TÁTICO", on_click=simular_recebimento_alerta, use_container_width=True)

    # --- Rodapé da Barra Lateral ---
    st.sidebar.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8rem;'>"
        "<strong>CARTIO - Simulador Logístico</strong><br>"
        "Versão: 4.1<br>"
        "Desenvolvido por: <strong>Wagner Calazans</strong>"
        "</div>", 
        unsafe_allow_html=True
    )

    # --- Back-end de Roteamento ---
    try:
        rota_segura = nx.shortest_path(st.session_state.G_atual, source=origem_selecionada, target=destino_selecionado, weight='peso_tatico')
        distancia_rota_m = sum([st.session_state.G_atual[rota_segura[i]][rota_segura[i+1]][0].get('length', 0) for i in range(len(rota_segura)-1)])
        tempo_estimado_min = (distancia_rota_m / 1000) / 40 * 60
        rota_interpolada = interpolar_rota(G, rota_segura, num_pontos=1000)
    except nx.NetworkXNoPath:
        rota_segura, rota_interpolada, distancia_rota_m, tempo_estimado_min = [], [], 0, 0

    posicoes_comboio = []
    tem_sinal_agora = False

    if rota_interpolada:
        idx_lider = int((progresso_missao / 100.0) * (len(rota_interpolada) - 1))
        espacamento_tatico = 15
        for i in range(4):
            idx_carro = max(0, idx_lider - (i * espacamento_tatico))
            posicoes_comboio.append(rota_interpolada[idx_carro])
            
        lon_lider, lat_lider = posicoes_comboio[0]
        tem_sinal_agora = any(haversine(lon_lider, lat_lider, gw["coords"][0], gw["coords"][1]) <= gw["radius"] for gw in GATEWAYS_LORAWAN)

    # --- Painel de Métricas (Top Metrics) ---
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        status_html = "<span class='status-online'>SISTEMA ONLINE</span>" if tem_sinal_agora else "<span class='status-offline'>SINAL DEGRADADO</span>"
        st.markdown(f"**STATUS LORAWAN:**<br><br>{status_html}", unsafe_allow_html=True)
    with m2:
        st.metric("DISTÂNCIA DA ROTA (KM)", f"{distancia_rota_m / 1000:.2f}")
    with m3:
        st.metric("TEMPO ESTIMADO (MIN)", f"{tempo_estimado_min:.1f}")
    with m4:
        num_obstaculos = 1 if st.session_state.alerta_acionado else 0
        st.metric("OBSTÁCULOS DETECTADOS", f"{num_obstaculos}")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Renderização Cinematográfica (PyDeck) ---
    layers = []

    # Gateways LoRaWAN
    gw_data = [{"coords": gw["coords"], "radius": gw["radius"]} for gw in GATEWAYS_LORAWAN]
    layer_gateways = pdk.Layer(
        "ScatterplotLayer",
        data=gw_data,
        get_position="coords",
        get_radius="radius",
        get_fill_color=[18, 72, 76, 50], # #12484C com transparência
        get_line_color=[43, 117, 116, 150], # #2B7574
        lineWidthMinPixels=2,
        pickable=True
    )
    layers.append(layer_gateways)

    # Rota Traçada
    if rota_interpolada:
        cor_rota = [43, 117, 116, 255] if st.session_state.alerta_acionado else [150, 150, 150, 200]
        layer_rota = pdk.Layer(
            "PathLayer",
            data=[{"path": rota_interpolada}],
            get_path="path",
            get_color=cor_rota,
            width_scale=20,
            width_min_pixels=4,
            get_width=5
        )
        layers.append(layer_rota)

    # Comboio de Veículos
    if posicoes_comboio:
        comboio_data = [{"coords": pos} for pos in posicoes_comboio]
        layer_comboio = pdk.Layer(
            "ScatterplotLayer",
            data=comboio_data,
            get_position="coords",
            get_radius=30,
            get_fill_color=[226, 226, 224, 255], 
            get_line_color=[14, 41, 49, 255],
            radius_min_pixels=6,
            radius_max_pixels=15,
            lineWidthMinPixels=2
        )
        layers.append(layer_comboio)

    # Ameaça / Bloqueio
    if st.session_state.ponto_bloqueio is not None:
        blq_lat, blq_lon = st.session_state.ponto_bloqueio
        layer_bloqueio = pdk.Layer(
            "ScatterplotLayer",
            data=[{"coords": [blq_lon, blq_lat]}],
            get_position="coords",
            get_radius=60,
            get_fill_color=[134, 18, 17, 220], # Grená
            radius_min_pixels=8,
            stroked=True,
            lineWidthMinPixels=2,
            get_line_color=[255, 255, 255, 255]
        )
        layers.append(layer_bloqueio)

    cam_lon, cam_lat = posicoes_comboio[0] if posicoes_comboio else (-43.195, -22.898)
    view_state = pdk.ViewState(latitude=cam_lat, longitude=cam_lon, zoom=14, pitch=50, bearing=0)

    # O PyDeck sempre renderiza mapas escuros pois é um dashboard tático
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=pdk.map_styles.DARK,
        tooltip={"text": "Localização Tática"}
    )

    st.pydeck_chart(r)


# ==========================================
# ABA 2: ARQUITETURA DE BORDA (VISÃO COMPUTACIONAL)
# ==========================================
with tab_arquitetura:
    
    st.markdown("### Integração de Visão Computacional de Borda e Roteamento Dinâmico")
    
    # Contexto delimitado conforme exigido pelo usuário
    st.info("""
    **O Escopo da Simulação (CARTIO):**
    O processo tem início com a captura de imagens pela câmera veicular e o processamento local (Edge) pelo modelo **YOLOv5**, que extrai as coordenadas da anomalia na pista. Em seguida, este pacote georreferenciado é transmitido via Rádio LoRaWAN para o Centro de Controle (CCO). Por fim, o servidor recebe a notificação, recalcula as rotas de emergência utilizando algoritmos em grafos (NetworkX) e plota a atualização tática no painel interativo.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Diagrama Lógico da Solução")
        
        # Diagrama Mermaid super detalhado baseado na imagem do usuário
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
        O simulador CARTIO serve como fundação de testes para melhorias contínuas. A arquitetura está preparada para as seguintes evoluções de pesquisa:
        
        - **Autocorrelação Espacial (PySAL / LISA):**
          Automatizar a identificação de *hotspots* regionais no GeoPackage, saindo da análise de pontos isolados para o mapeamento e clusterização de áreas de risco sistêmico na rede viária.
          
        - **Função de Custo Expandida (OSMnx / NetworkX):**
          Implementar algoritmo que penaliza o peso tático não apenas da rua bloqueada, mas de todas as vias vizinhas no grafo do OpenStreetMap, forçando desvios preventivos mais amplos e seguros.
          
        - **Validação de Redes (CARTIO / LoRaWAN):**
          Avaliar o impacto direto do atraso de rede (delay de propagação) e do volume de dados de múltiplas viaturas simultâneas na velocidade de triagem criptográfica antes da plotagem final das rotas no GeoPandas.
        """)
