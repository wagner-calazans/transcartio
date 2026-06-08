# Sistema Tático de Suporte à Decisão Logística (CARTIO)

![CARTIO Banner](https://img.shields.io/badge/Status-Ativo-success) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B)

## Visão Geral
O **CARTIO** é um simulador de Roteamento Dinâmico de Vias sob cenários de infraestrutura de rede degradada (Blecautes de Internet). Este painel (Dashboard Tático) é a camada final de visualização do CCO (Centro de Controle Operacional), desenhado para reagir de forma autônoma a ameaças físicas nas vias mapeadas.

A inovação principal do projeto baseia-se no uso de Inteligência Artificial na borda (**YOLOv5** embarcado em Raspberry Pi) em viaturas táticas que identificam o bloqueio da via (ex: alagamentos, crateras, barricadas) e enviam as coordenadas geográficas através de ondas de rádio **LoRaWAN** (915 MHz), totalmente independentes da internet comercial.

Ao receber o pacote criptografado pelo rádio, o servidor CARTIO recalcula a rota do comboio utilizando **NetworkX e OSMnx**, garantindo que as viaturas de suprimento logístico não entrem na área de risco.

## 🛠️ Instalação e Execução

### 1. Clonando o Repositório
```bash
git clone https://github.com/wagner-calazans/transcartio.git
cd transcartio
```

### 2. Instalando as Dependências
É altamente recomendado criar um ambiente virtual (venv):
```bash
python -m venv venv
source venv/bin/activate  # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Rodando o Simulador
Execute o comando padrão do Streamlit na pasta raiz do projeto:
```bash
streamlit run app_dashboard_tatico.py
```
O simulador será aberto automaticamente no seu navegador em `http://localhost:8501`.

## 🎮 Guia de Uso da Interface (UI)

O sistema possui uma interface responsiva focada no modo *Deep Dark* para CCOs. 
A navegação é centralizada em duas abas principais:

### Aba 1: Simulador Operacional (PyDeck 3D)
- **Barra Lateral**: Controle o Ponto de Origem e Destino do Comboio Tático.
- **Slider "Progresso da Missão"**: Mova a barra para fazer com que os 4 veículos laranjas se desloquem fisicamente pela rota calculada no painel 3D.
- **Física LoRaWAN**: Repare nos **Círculos Verdes Translúcidos** sobre o mapa. Eles são a representação física do raio de alcance dos Gateways de rádio (ex: Maracanã e Caju).
- **Injeção de Ameaça**: Quando você envia um cenário de caos (ex: Alagamento), o sistema verifica se a viatura está dentro do raio verde (Online). Caso esteja fora, a transmissão falha (Offline). Se for bem-sucedida, a rota pisca em azul ciano e desvia da área vermelha!
- **Modo Demo (Cinemático)**: Existe um botão "Iniciar Demo Automatizada". Ao clicar nele, você pode soltar o mouse. O sistema rodará uma simulação interativa como cena de filme, testando a resiliência da rede automaticamente.

### Aba 2: Arquitetura de Borda (Edge AI)
Apresenta a fundamentação teórica que liga o asfalto à nuvem. Mostra um diagrama estrutural de como o sensor da Câmera (Raspberry V3) captura as imagens, passa pelo YOLOv5 (conversão via libonnxruntime) e dispara a instrução na porta Serial para a Antena Reyax.

---
**Instituição**: IME - Instituto Militar de Engenharia  
**Pesquisador**: Wagner Calazans (2026)
