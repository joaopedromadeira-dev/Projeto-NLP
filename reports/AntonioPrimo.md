## Diário de bordo projeto NLP

### 2025-03-05

a) Adicionado PDF candidato : Claúsulas Gerais Contrato Abertura Cheque Especial Banco do Brasil;
b) Sumarização orientações Projeto Integrador;
c) Sumarização orientações Projeto NLP.


### 2025-03-06

Procurar bibliotecas para extração de PDFs.

Utilizado o pymupdf.

Adptado código da biblioteca Python disponível em: https://pymupdf.readthedocs.io/en/latest/the-basics.html
import pymupdf  # ou fitz (dependendo da instalação)

# Caminho do arquivo (substitua pelo seu caminho, se necessário)
pdf_path = r"/workspaces/hf_tests/CLÁUSULAS GERAIS DO CONTRATO DE ABERTURA DE CRÉDITO EM CONTA CORRENTE - LIMITE ESPECIAL DA CONTA.pdf"

doc = pymupdf.open(pdf_path)
texto_completo = ""

for pagina_num in range(len(doc)):
    pagina = doc[pagina_num]
    texto = pagina.get_text()
    texto_completo += f"\n--- Página {pagina_num+1} ---\n"
    texto_completo += texto

# Salva o texto extraído em um arquivo .txt
with open("documento_extraido.txt", "w", encoding="utf-8") as f:
    f.write(texto_completo)

print("Texto extraído com sucesso! Verifique o arquivo 'documento_extraido.txt'.")

Extraido o texto do documento CLÁUSULAS GERAIS DO CONTRATO DE ABERTURA DE CRÉDITO EM CONTA CORRENTELIMITE ESPECIAL DA CONTA. Feita a limpeza do texto e ajustada a formatação.


### 2025-03-09
Gold Standard

Feito Brasinstorming de perguntas e respostas(feita a marcaçao do paragrafos) com base no PDF TCC, capitulo conceitos básicos do IPV6. 

Perguntas:


1) Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast, considerando aspectos como eficiência de banda, escalabilidade e aplicações típicas? 
Como o multicast otimiza a transmissão de dados para múltiplos destinos simultaneamente, e em que cenários seu uso é mais indicado do que o unicast ou anycast?

Respostas: 
O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem parte
de um determinado endereço multicast. A vantagem que uma transmissão tem em relação ao unicast é a
economia de banda da rede, pois transmite um único pacote para vários nós ao mesmo tempo, o que não
ocorre com uma transmissão unicast, onde é transmitido um pacote para cada nó da rede. Algumas
transmissões de vídeos pela rede usam multicast, economizando assim a banda da rede - Pagina 22 - Primeiro paragrafo do item 2.3.2


2) Além da forma preferencial de representação dos endereços IPv6, quais são os métodos de abreviação permitidos? 
Explique as regras para supressão de zeros à esquerda e para substituição de sequências de blocos nulos por "::", incluindo as limitações quanto ao uso múltiplo dessa abreviação para evitar ambiguidades.

Respostas: 
i) Um dos métodos utilizados na representação do endereçamento é de suprimir os zeros da esquerda de cada
parte de 16 bits [HID 98]. Para exemplificar este método, usaremos como exemplo “1.2.2”. Aplicando o
método de representação, o exemplo ficaria da seguinte forma:
1080:8:0:800:0:0:0:0.;

ii) Outra forma de simplificar é suprimir as seqüências de 16 bits com valores iguais a zero. Assim, os
exemplos 1.2.2 e 1.2.3 ficariam respectivamente 1080:0008:0000:0800:: e 3FFE:2B00:0100:0107::0001.
Deve-se ter o cuidado de suprimir somente uma seqüência de 16 bits. Ou seja, considerando o exemplo
1.2.2, ele não poderia ficar assim 1080:0008::0800::. Pode se ver que ele tem duas vezes o sinal de dois
pontos juntos o que seria inválido. Somente se pode ter uma única vez os sinais de dois pontos juntos [HID
98].

página 17 - Segundo e terceiro paragrafo do item 2.2

3) Qual foi a principal motivação para o desenvolvimento do protocolo IPv6, considerando o contexto de esgotamento dos endereços IPv4 na década de 1990?
Descreva as etapas intermediárias, como o CIDR, e o papel da IETF na criação do IPng (IPv6) para solucionar as limitações de endereçamento da Internet.

Resposta: Desde 1990 já se sabia da necessidade de aumentar o endereçamento dos números IPs, devido ao grande
aumento que estava ocorrendo na Internet e que em pouco tempo se esgotariam os números IPs. Este
problema eminente foi resolvido em duas etapas, a primeira foi a redistribuição dos endereços através da
proposta do Classless Inter-Domain Routing (CIDR), especificado na RFC 1518 e 1519. Isto deu um tempo
a mais para a criação de uma nova proposta de endereçamento [SAL 00]. A segunda etapa foi a criação de
um novo protocolo que aumentaria os endereços IPs. Então a IETF (Internet Engineering Task Force) criou
um grupo de trabalho para desenvolver o novo protocolo chamado inicialmente de IPng (Internet Protocol
New Generation), o qual iria substituir o protocolo atual que passou a ser denominado IPv4 (Internet
Protocol version 4). Então, em 1994, após várias discussões e revisões de propostas para o novo protocolo,
o grupo de trabalho do IPng decidiu que caminho seguir. A proposta escolhida foi o SIPP (Simple IP Plus)
[SIL 97] - Página 16 - quarto paragrafo.


4) Quais campos do cabeçalho IPv4 foram eliminados ou modificados no cabeçalho base do IPv6 para simplificar o processamento pelos roteadores? 
Cite os campos que permaneceram com a mesma função e explique como essa redução contribui para a eficiência do protocolo.

Respota:Portanto, dos quatorze campos que existem na
versão anterior, passou-se a ter oito campos na nova versão do IP. Somente três campos continuam com a
mesma denominação e função, que são: version, source address e destination address. Alguns campos
como, por exemplo, o total length do IPv4 foi substituído por payload length e teve uma pequena mudança
como será visto mais adiante - página 26 - primeiro paragrafo.

5) Quais são as regras fundamentais para a implementação e uso correto do campo Flow Label no cabeçalho IPv6?
Detalhe as condições que devem ser seguidas por hosts e roteadores, incluindo o tratamento de valores nulos, a consistência de parâmetros para pacotes de um mesmo fluxo e a faixa de valores permitidos para identificar fluxos de dados.

Resposta:  Existem três regras a serem cumpridas quando é implementado o uso do flow label, estas regras são [STA
00]:

 - Host e roteadores que não suportam o flow label devem, ao criar o pacote, atribuir o valor zero para
o campo, não modificar o campo quando for repassar o pacote e ignorar o campo quando receber o
pacote.
 - Os pacotes criados por um nó com o mesmo valor e não zero do flow label devem ter o mesmo
endereço destino e origem, e se for o caso, o Hop-by-hop Options Header e Routing Header iguais
(estes dois cabeçalhos serão apresentados no capitulo dois).
 - Para obter um valor flow label para um fluxo de dados, deve ser escolhido um valor na faixa entre 1
e (220 – 1) e não se deve reusar o mesmo valor do flow label para um novo fluxo de dados quando
terminar o tempo de vida do primeiro. O valor zero é reservado para informar que o campo não está
sendo usado. Página 26 e 27 do item 2.4 - terceiro paragrafo.

6) Como o campo Payload Length do IPv6 difere do campo Total Length do IPv4 em termos de escopo de contagem? Qual é a vantagem prática de excluir o cabeçalho principal do cálculo do tamanho do pacote?

Resposta: O campo Payload Length é equivalente ao campo total length (tamanho total) no IPv4. A diferença entre
eles é que, por definição, o payload length significa somente o tamanho de dados em octetos após o
cabeçalho, excluindo o tamanho do cabeçalho. Isso não ocorria no IPv4, que contabilizava tudo, inclusive o
cabeçalho [HUI 00]. 4 paragrafo - pagina 27

7) Qual é a função do campo Hop Limit no IPv6 e como ele evita que pacotes fiquem indefinidamente em loop na rede? Explique o mecanismo de decremento e a ação tomada quando o valor atinge zero.
Resposta: O campo Hop Limit é um inteiro positivo de 8 bits de tamanho. Sua funcionalidade é de saber quando o
pacote deve ser descartado por um roteador, assim, o pacote não ficará vagando eternamente pela rede, se
não for encontrado o seu destino. Funciona da seguinte forma: cada vez que o pacote é passado adiante
por um roteador, e seu valor é decrementado de um. Quando o valor for igual a zero, o pacote é descartado
[SAL 00]. sexto paragrafo - pagina 27

8) Qual é o tamanho dos campos de endereço de origem e destino no IPv6 e qual a principal consequência desse aumento em relação à capacidade de endereçamento da Internet?
Resposta: Os campos source address e destination address são, respectivamente, os endereços de origem e destino
do pacote. Cada um destes campos possui o tamanho de 128 bits [DEE 98]. sétimo paragrafo - pagina 27

9) O que caracteriza um endereço anycast e em que situações práticas ele é mais vantajoso que o unicast ou multicast? Dê um exemplo de aplicação envolvendo servidores de nomes e explique como o roteamento decide qual destino será alcançado.
Resposta: O conceito de endereço anycast é mais recente do que o unicast e o multicast. Pode-se dizer que o anycast
é uma mistura do unicast e do multicast. A finalidade deste endereço pode ser exemplificada numa situação,
onde se deseja requisitar um servidor de nomes de uma rede. A máquina requisitante irá enviar um pacote
para um determinado endereço anycast, que todos os servidores de nome possuem. O roteador se
encarregará de entregar o pacote ao servidor mais próximo da máquina que requisitou [HUI 98]. pagina 27 - primeiro paragrafo do item 2.3.3

10) Quais são os dois tipos de endereços IPv6 utilizados para compatibilidade com redes IPv4? Explique a estrutura de cada um (destaque os bits de alta e baixa ordem) e forneça exemplos de representação abreviada, indicando em que contexto cada tipo é aplicado.
Resposta: 
Endereço de translação para IPV4:
Como existe a idéia do IPv6 substituir aos poucos as redes IPv4, foi criado um mecanismo para fazer o
tunelamento dinamicamente de pacotes IPv6 para pacotes IPv4. Os nós que usam esta técnica têm um
endereço unicast especial, que carrega nos 32-bits de baixa prioridade um endereço IPv4. Os restantes dos
96 bits são representados por zeros. Esta técnica é chamada de IPv4, compatível com endereço IPv6 [HID
98]. O endereço 0:0:0:0:0:0:10.16.169.1 (ou ::10.16.169.1) é um exemplo dessa técnica.
Existe também um segundo tipo de endereço de translação que é o mapeamento do IPv4 para endereço
IPv6. Ou seja, serve para representar nós que não suportam endereços IPv6. Foi definido que este tipo de
endereço possui os 80 primeiros bits de alta prioridade com valores zero. Os 16 bits após têm o valor um e
os 32 bits restantes representam um endereço IPv4 [HID 98]. Um exemplo deste tipo de mapeamento é
representado pelo endereço 0:0:0:0:0:FFFF:10.16.169.1, ou simplificando ::FFFF:10.16.169.1. - página 20 - segundo paragrafo.

### 2025-03-10

Experimentados modelos do sentencetransformes:

1) Modelos de Busca Semântica ("multi-qa-mpnet-base-cos-v1"): 
Este é um modelo de transformação de sentenças : ele mapeia sentenças e parágrafos para um espaço vetorial denso de 768 dimensões e foi projetado para busca semântica . 
Ele foi treinado com 215 milhões de pares (pergunta, resposta) de diversas fontes. Para uma introdução à busca semântica, consulte: SBERT.net - Busca Semântica

####
from sentence_transformers import SentenceTransformer, util

query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

#Load the model
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')

#Encode query and documents
query_emb = model.encode(query)
doc_emb = model.encode(docs)

#Compute dot score between query and all document embeddings
scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

#Combine docs & scores
doc_score_pairs = list(zip(docs, scores))

#Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

#Output passages & scores
for doc, score in doc_score_pairs:
    print(score, doc)
###
Resultado: 
Métricas de avaliação:
Precision@1: 0.7000
Precision@3: 0.8000
Precision@5: 0.9000
MRR: 0.7893

--- Exemplo de consulta ---
0.7126	Endereço de translação para IPV4: Como existe a idéia do IPv6 substituir aos poucos as redes IPv4, f...
0.6231	Desde 1990 já se sabia da necessidade de aumentar o endereçamento dos números IPs, devido ao grande ...
0.5616	Portanto, dos quatorze campos que existem na versão anterior, passou-se a ter oito campos na nova ve...


#####
2) Modelos de controle de qualidade múltiplo

multi-qa-mpnet-base-dot-v1

Este é um modelo de transformação de sentenças : ele mapeia sentenças e parágrafos para um espaço vetorial denso de 768 dimensões e foi projetado para busca semântica .
Ele foi treinado com 215 milhões de pares (pergunta, resposta) de diversas fontes. Para uma introdução à busca semântica, consulte: SBERT.net - Busca Semântica

from sentence_transformers import SentenceTransformer, util

query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

# Load the model
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

# Encode query and documents
query_emb = model.encode(query)
doc_emb = model.encode(docs)

# Compute dot score between query and all document embeddings
scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

# Combine docs & scores
doc_score_pairs = list(zip(docs, scores))

# Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

# Output passages & scores
for doc, score in doc_score_pairs:
    print(score, doc)
###
Resultados
Métricas de avaliação (modelo dot-v1, similaridade por produto escalar):
Precision@1: 0.8000
Precision@3: 0.9000
Precision@5: 0.9000
MRR: 0.8500

--- Exemplo de consulta ---
25.9954	O conceito de endereço anycast é mais recente do que o unicast e o multicast. Pode-se dizer que o an...
19.2742	O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem p...
17.0688	Portanto, dos quatorze campos que existem na versão anterior, passou-se a ter oito campos na nova ve...

3) Modelos de Passagem MSMARCO
##
from sentence_transformers import SentenceTransformer, util

query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

#Load the model
model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')

#Encode query and documents
query_emb = model.encode(query)
doc_emb = model.encode(docs)

#Compute dot score between query and all document embeddings
scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

#Combine docs & scores
doc_score_pairs = list(zip(docs, scores))

#Sort by decreasing score
doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

#Output passages & scores
print("Query:", query)
for doc, score in doc_score_pairs:
    print(score, doc)

##
Resultados:
Métricas de avaliação (modelo msmarco-bert-base-dot-v5, similaridade por produto escalar):
Precision@1: 0.7000
Precision@3: 0.9000
Precision@5: 1.0000
MRR: 0.8200

### 2026-03-11
Primeiro contato com o langchain:
https://docs.langchain.com/oss/python/langchain/knowledge-base

### 2026-03-12
Carregar documento PDF:

####
import os
!pip install -qU langchain-community pypdf

from langchain_community.document_loaders import PyPDFLoader

def carregar_pdf(caminho_arquivo):
    """
    Carrega um documento PDF e retorna as páginas como documentos.

    Args:
        caminho_arquivo (str): Caminho para o arquivo PDF.

    Returns:
        list: Lista de documentos (páginas) carregados.
    """
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    loader = PyPDFLoader(caminho_arquivo)
    documentos = loader.load()
    return documentos

if __name__ == "__main__":
    # Exemplo: permite que o usuário digite o caminho ou use um padrão
    caminho_padrao = "../example_data/nke-10k-2023.pdf"

    # Opção 1: usar caminho fixo (como no código original)
    # caminho = caminho_padrao

    # Opção 2: perguntar ao usuário
    caminho = input("Digite o caminho do arquivo PDF (ou pressione Enter para usar o padrão): ").strip()
    if not caminho:
        caminho = caminho_padrao

    try:
        docs = carregar_pdf(caminho)
        print(f"Total de páginas carregadas: {len(docs)}")

        # Exibe a primeira página como exemplo
        if docs:
            print("\n--- Primeira página ---")
            print(docs[0].page_content[:500])  # primeiros 500 caracteres
            print(f"Metadados: {docs[0].metadata}")
    except Exception as e:
        print(f"Erro ao carregar o PDF: {e}")

####
Chunks:
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Caminho específicO
caminho_pdf = "/content/drive/MyDrive/UC15/TCC  versão 4.6.pdf"

# Verificar se o arquivo existe
if not os.path.exists(caminho_pdf):
    print(f"ERRO: Arquivo não encontrado em: {caminho_pdf}")
    print("Verifique se o Google Drive está montado e o caminho está correto.")
else:
    print(f"Arquivo encontrado: {caminho_pdf}")
    
    # Carregar o documento PDF
    loader = PyPDFLoader(caminho_pdf)
    docs = loader.load()
    
    print(f"Total de páginas carregadas: {len(docs)}")
    
    # Configurar e aplicar o splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True
    )
    
    all_splits = text_splitter.split_documents(docs)
    
    print(f"Total de chunks após divisão: {len(all_splits)}")
    
    # Opcional: mostrar informações sobre os primeiros chunks
    if all_splits:
        print("\n--- Primeiro chunk ---")
        print(f"Tamanho: {len(all_splits[0].page_content)} caracteres")
        print(f"Conteúdo (início): {all_splits[0].page_content[:200]}...")
        print(f"Metadados: {all_splits[0].metadata}")

###
EMBEDDINGS:
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from google.colab import drive

caminho_pdf = "/content/drive/MyDrive/UC15/TCC  versão 4.6.pdf"

# Carregar e dividir
docs = PyPDFLoader(caminho_pdf).load()
all_splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Criar embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)
vector_3 = embeddings.embed_query(all_splits[2].page_content)
vector_4 = embeddings.embed_query(all_splits[3].page_content)
vector_5 = embeddings.embed_query(all_splits[4].page_content)

print(f"Vetores gerados - Tamanho: {len(vector_1)}")
print(f"Primeiros 10 elementos: {vector_1[:10]}")
####
ARMAZENAMENTO EM ESPACO VETORIAL
*** In memory *** 
pip install -U "langchain-core"

from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents=all_splits)

print(f"IDs dos documentos adicionados: {ids[:5]}...")  # Mostra primeiros 5 IDs
print(f"Total de documentos adicionados: {len(ids)}")
print(f"Total de vetores no índice: {len(ids)}")

*** chroma Data base *** 

pip install -qU langchain-chroma
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    collection_name="tcc_uc15_collection",
    persist_directory="/content/drive/MyDrive/UC15/chroma_langchain_db"
)

print(f"Vector store criado com {vector_store._collection.count()} documentos")


######
BUSCA POR SIMILIARIDADE
# Realizar uma busca de similaridade
query = "O que é o protocolo IPV6?"
docs_busca = vector_store.similarity_search(query, k=3)

print("\nResultados da busca:")
for i, doc in enumerate(docs_busca):
    print(f"\n--- Resultado {i+1} ---")
    print(f"Conteúdo: {doc.page_content[:200]}...")
    print(f"Metadados: {doc.metadata}")

results = vector_store.similarity_search_with_score("Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast, considerando aspectos como eficiência de banda, escalabilidade e aplicações típicas")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)

##### 2026-03-16

1. Instalação Firefox VDI 
2. Elaboração do código para  combinação de busca semântica (bi-encoder) com re-ranking (cross-encoder):
    a. Carrega seu conjunto específico de 10 perguntas e respostas sobre IPv6

    b. Seleciona aleatoriamente 3 perguntas usando random.sample()

   c.  Mantém a funcionalidade de busca semântica com bi-encoder

   d.  Aplica re-ranking com cross-encoder para melhorar os resultados

    e. Mostra a resposta correta esperada e indica se ela aparece nos resultados

    f. Permite testar perguntas personalizadas ao final


"""
### 2026-03-17

Finalizacao do codigo que executa os seguintes passos (Retrieve & Re-Rank para perguntas sobre IPv6
):

1. Carrega um conjunto de 10 perguntas e respostas sobre IPv6
2. Seleciona aleatoriamente 3 perguntas para busca
3. Usa um bi-encoder para busca semântica inicial
4. Reordena os resultados com um cross-encoder para melhor precisão

RETRIVE E RE-RANK
# !pip install -U sentence-transformers rank_bm25

import random
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")

# Configuração dos modelos
bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
bi_encoder.max_seq_length = 256  # Truncar passagens longas para 256 tokens
top_k = 10  # Número de passagens a recuperar com o bi-encoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# Conjunto de perguntas e respostas sobre IPv6
qa_pairs = [
    {
        "question": "Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast?",
        "answer": """O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem parte
de um determinado endereço multicast. A vantagem que uma transmissão tem em relação ao unicast é a
economia de banda da rede, pois transmite um único pacote para vários nós ao mesmo tempo, o que não
ocorre com uma transmissão unicast, onde é transmitido um pacote para cada nó da rede. Algumas
transmissões de vídeos pela rede usam multicast, economizando assim a banda da rede - Pagina 22 - Primeiro paragrafo do item 2.3.2"""
    },
    {
        "question": "Além da forma preferencial de representação dos endereços IPv6, quais são os métodos de abreviação permitidos?",
        "answer": """i) Um dos métodos utilizados na representação do endereçamento é de suprimir os zeros da esquerda de cada
parte de 16 bits [HID 98]. Para exemplificar este método, usaremos como exemplo “1.2.2”. Aplicando o
método de representação, o exemplo ficaria da seguinte forma:
1080:8:0:800:0:0:0:0.;

ii) Outra forma de simplificar é suprimir as seqüências de 16 bits com valores iguais a zero. Assim, os
exemplos 1.2.2 e 1.2.3 ficariam respectivamente 1080:0008:0000:0800:: e 3FFE:2B00:0100:0107::0001.
Deve-se ter o cuidado de suprimir somente uma seqüência de 16 bits. Ou seja, considerando o exemplo
1.2.2, ele não poderia ficar assim 1080:0008::0800::. Pode se ver que ele tem duas vezes o sinal de dois
pontos juntos o que seria inválido. Somente se pode ter uma única vez os sinais de dois pontos juntos [HID
98].

página 17 - Segundo e terceiro paragrafo do item 2.2"""
    },
    {
        "question": "Qual foi a principal motivação para o desenvolvimento do protocolo IPv6, considerando o contexto de esgotamento dos endereços IPv4 na década de 1990?",
        "answer": """Desde 1990 já se sabia da necessidade de aumentar o endereçamento dos números IPs, devido ao grande
aumento que estava ocorrendo na Internet e que em pouco tempo se esgotariam os números IPs. Este
problema eminente foi resolvido em duas etapas, a primeira foi a redistribuição dos endereços através da
proposta do Classless Inter-Domain Routing (CIDR), especificado na RFC 1518 e 1519. Isto deu um tempo
a mais para a criação de uma nova proposta de endereçamento [SAL 00]. A segunda etapa foi a criação de
um novo protocolo que aumentaria os endereços IPs. Então a IETF (Internet Engineering Task Force) criou
um grupo de trabalho para desenvolver o novo protocolo chamado inicialmente de IPng (Internet Protocol
New Generation), o qual iria substituir o protocolo atual que passou a ser denominado IPv4 (Internet
Protocol version 4). Então, em 1994, após várias discussões e revisões de propostas para o novo protocolo,
o grupo de trabalho do IPng decidiu que caminho seguir. A proposta escolhida foi o SIPP (Simple IP Plus)
[SIL 97] - Página 16 - quarto paragrafo."""
    },
    {
        "question": "Quais campos do cabeçalho IPv4 foram eliminados ou modificados no cabeçalho base do IPv6 para simplificar o processamento pelos roteadores?",
        "answer": """Portanto, dos quatorze campos que existem na
versão anterior, passou-se a ter oito campos na nova versão do IP. Somente três campos continuam com a
mesma denominação e função, que são: version, source address e destination address. Alguns campos
como, por exemplo, o total length do IPv4 foi substituído por payload length e teve uma pequena mudança
como será visto mais adiante - página 26 - primeiro paragrafo."""
    },
    {
        "question": "Quais são as regras fundamentais para a implementação e uso correto do campo Flow Label no cabeçalho IPv6?",
        "answer": """Existem três regras a serem cumpridas quando é implementado o uso do flow label, estas regras são [STA
00]:

 - Host e roteadores que não suportam o flow label devem, ao criar o pacote, atribuir o valor zero para
o campo, não modificar o campo quando for repassar o pacote e ignorar o campo quando receber o
pacote.
 - Os pacotes criados por um nó com o mesmo valor e não zero do flow label devem ter o mesmo
endereço destino e origem, e se for o caso, o Hop-by-hop Options Header e Routing Header iguais
(estes dois cabeçalhos serão apresentados no capitulo dois).
 - Para obter um valor flow label para um fluxo de dados, deve ser escolhido um valor na faixa entre 1
e (220 – 1) e não se deve reusar o mesmo valor do flow label para um novo fluxo de dados quando
terminar o tempo de vida do primeiro. O valor zero é reservado para informar que o campo não está
sendo usado. Página 26 e 27 do item 2.4 - terceiro paragrafo."""
    },
    {
        "question": "Como o campo Payload Length do IPv6 difere do campo Total Length do IPv4 em termos de escopo de contagem?",
        "answer": """O campo Payload Length é equivalente ao campo total length (tamanho total) no IPv4. A diferença entre
eles é que, por definição, o payload length significa somente o tamanho de dados em octetos após o
cabeçalho, excluindo o tamanho do cabeçalho. Isso não ocorria no IPv4, que contabilizava tudo, inclusive o
cabeçalho [HUI 00]. 4 paragrafo - pagina 27"""
    },
    {
        "question": "Qual é a função do campo Hop Limit no IPv6 e como ele evita que pacotes fiquem indefinidamente em loop na rede?",
        "answer": """O campo Hop Limit é um inteiro positivo de 8 bits de tamanho. Sua funcionalidade é de saber quando o
pacote deve ser descartado por um roteador, assim, o pacote não ficará vagando eternamente pela rede, se
não for encontrado o seu destino. Funciona da seguinte forma: cada vez que o pacote é passado adiante
por um roteador, e seu valor é decrementado de um. Quando o valor for igual a zero, o pacote é descartado
[SAL 00]. sexto paragrafo - pagina 27"""
    },
    {
        "question": "Qual é o tamanho dos campos de endereço de origem e destino no IPv6 e qual a principal consequência desse aumento em relação à capacidade de endereçamento da Internet?",
        "answer": """Os campos source address e destination address são, respectivamente, os endereços de origem e destino
do pacote. Cada um destes campos possui o tamanho de 128 bits [DEE 98]. sétimo paragrafo - pagina 27"""
    },
    {
        "question": "O que caracteriza um endereço anycast e em que situações práticas ele é mais vantajoso que o unicast ou multicast?",
        "answer": """O conceito de endereço anycast é mais recente do que o unicast e o multicast. Pode-se dizer que o anycast
é uma mistura do unicast e do multicast. A finalidade deste endereço pode ser exemplificada numa situação,
onde se deseja requisitar um servidor de nomes de uma rede. A máquina requisitante irá enviar um pacote
para um determinado endereço anycast, que todos os servidores de nome possuem. O roteador se
encarregará de entregar o pacote ao servidor mais próximo da máquina que requisitou [HUI 98]. pagina 27 - primeiro paragrafo do item 2.3.3"""
    },
    {
        "question": "Quais são os dois tipos de endereços IPv6 utilizados para compatibilidade com redes IPv4?",
        "answer": """Endereço de translação para IPV4:
Como existe a idéia do IPv6 substituir aos poucos as redes IPv4, foi criado um mecanismo para fazer o
tunelamento dinamicamente de pacotes IPv6 para pacotes IPv4. Os nós que usam esta técnica têm um
endereço unicast especial, que carrega nos 32-bits de baixa prioridade um endereço IPv4. Os restantes dos
96 bits são representados por zeros. Esta técnica é chamada de IPv4, compatível com endereço IPv6 [HID
98]. O endereço 0:0:0:0:0:0:10.16.169.1 (ou ::10.16.169.1) é um exemplo dessa técnica.
Existe também um segundo tipo de endereço de translação que é o mapeamento do IPv4 para endereço
IPv6. Ou seja, serve para representar nós que não suportam endereços IPv6. Foi definido que este tipo de
endereço possui os 80 primeiros bits de alta prioridade com valores zero. Os 16 bits após têm o valor um e
os 32 bits restantes representam um endereço IPv4 [HID 98]. Um exemplo deste tipo de mapeamento é
representado pelo endereço 0:0:0:0:0:FFFF:10.16.169.1, ou simplificando ::FFFF:10.16.169.1. - página 20 - segundo paragrafo."""
    }
]

# Preparar o corpus de respostas
answers = [qa["answer"] for qa in qa_pairs]

print(f"Total de perguntas/respostas carregadas: {len(answers)}")

# Codificar todas as respostas para busca semântica
print("\nCodificando respostas para busca semântica...")
corpus_embeddings = bi_encoder.encode(answers, convert_to_tensor=True, show_progress_bar=True)

def search(query, expected_answer_index=None):
    """
    Função de busca que encontra respostas relevantes para uma pergunta
    """
    print(f"\n{'='*60}")
    print(f"Pergunta: {query}")
    print(f"{'='*60}")
    
    # Se temos o índice da resposta esperada, mostrar ela primeiro
    if expected_answer_index is not None:
        print("\n✅ Resposta correta esperada (primeiro parágrafo):")
        preview = answers[expected_answer_index][:200] + "..."
        print(f"   {preview}")
    
    ##### Busca Semântica #####
    # Codificar a pergunta usando o bi-encoder
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    if torch.cuda.is_available():
        question_embedding = question_embedding.cuda()
    
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]
    
    # Adicionar scores da busca semântica inicial
    for hit in hits:
        hit['retrieval-score'] = hit['score']
    
    ##### Re-Ranking com Cross-Encoder #####
    cross_inp = [[query, answers[hit["corpus_id"]]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)
    
    for idx in range(len(cross_scores)):
        hits[idx]["cross-score"] = cross_scores[idx]
    
    # Resultados da busca semântica inicial
    print(f"\n📊 Top-{min(3, len(hits))} resultados da Busca Semântica Inicial:")
    hits_retrieval = sorted(hits, key=lambda x: x["retrieval-score"], reverse=True)
    for i, hit in enumerate(hits_retrieval[:3]):
        preview = answers[hit["corpus_id"]][:100].replace("\n", " ") + "..."
        print(f"   {i+1}. Score: {hit['retrieval-score']:.3f} | {preview}")
        if expected_answer_index is not None and hit["corpus_id"] == expected_answer_index:
            print("      ⭐ ESTA É A RESPOSTA CORRETA!")
    
    # Resultados após re-ranking
    print(f"\n🎯 Top-{min(3, len(hits))} resultados após Re-ranking (Cross-Encoder):")
    hits_reranked = sorted(hits, key=lambda x: x["cross-score"], reverse=True)
    for i, hit in enumerate(hits_reranked[:3]):
        preview = answers[hit["corpus_id"]][:100].replace("\n", " ") + "..."
        print(f"   {i+1}. Score: {hit['cross-score']:.3f} | {preview}")
        if expected_answer_index is not None and hit["corpus_id"] == expected_answer_index:
            print("      ⭐ ESTA É A RESPOSTA CORRETA!")
    
    # Verificar se a resposta correta está no top-3 após re-ranking
    if expected_answer_index is not None:
        top3_ids = [hit["corpus_id"] for hit in hits_reranked[:3]]
        if expected_answer_index in top3_ids:
            print(f"\n✅ SUCESSO: A resposta correta está no Top-3 após re-ranking!")
        else:
            print(f"\n❌ A resposta correta NÃO está no Top-3 após re-ranking.")
            print(f"   Posição da resposta correta: {[hit['corpus_id'] for hit in hits_reranked].index(expected_answer_index) + 1}ª")

# Selecionar 3 perguntas aleatórias
print("\n🎲 Selecionando 3 perguntas aleatórias do conjunto...")
selected_indices = random.sample(range(len(qa_pairs)), 3)
print(f"Índices selecionados: {selected_indices}")

# Executar busca para cada pergunta selecionada
for idx in selected_indices:
    question = qa_pairs[idx]["question"]
    search(question, expected_answer_index=idx)
    print("\n" + "="*60 + "\n")

# Opcional: Teste com uma pergunta personalizada
print("\n🔍 Teste com pergunta personalizada:")
search("Como funciona o anycast no IPv6?")
=====


### 2026-03-18

1.Leitura e compreensão dos colab da aula de hoje:
a)Código de referência do 2-stage IR usando a API do langchain_classic.
b)_código de referência do 2-stage IR refatorado para (segundo o LLM).

### 2026-03-19

Ausente, reunião escolar.

### 2026-03-20

Criação do código e testes na VM (legacy)


CUSTOMIZE O PRIMEIRO CODIGO(RETRIVE) PARA O 2(LEGACY): RETRIVE E RE-RANK
# !pip install -U sentence-transformers rank_bm25

import random
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")

# Configuração dos modelos
bi_encoder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
bi_encoder.max_seq_length = 256  # Truncar passagens longas para 256 tokens
top_k = 10  # Número de passagens a recuperar com o bi-encoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

# Conjunto de perguntas e respostas sobre IPv6
qa_pairs = [
    {
        "question": "Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast?",
        "answer": """O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem parte
de um determinado endereço multicast. A vantagem que uma transmissão tem em relação ao unicast é a
economia de banda da rede, pois transmite um único pacote para vários nós ao mesmo tempo, o que não
ocorre com uma transmissão unicast, onde é transmitido um pacote para cada nó da rede. Algumas
transmissões de vídeos pela rede usam multicast, economizando assim a banda da rede - Pagina 22 - Primeiro paragrafo do item 2.3.2"""
    },
    {
        "question": "Além da forma preferencial de representação dos endereços IPv6, quais são os métodos de abreviação permitidos?",
        "answer": """i) Um dos métodos utilizados na representação do endereçamento é de suprimir os zeros da esquerda de cada
parte de 16 bits [HID 98]. Para exemplificar este método, usaremos como exemplo “1.2.2”. Aplicando o
método de representação, o exemplo ficaria da seguinte forma:
1080:8:0:800:0:0:0:0.;

ii) Outra forma de simplificar é suprimir as seqüências de 16 bits com valores iguais a zero. Assim, os
exemplos 1.2.2 e 1.2.3 ficariam respectivamente 1080:0008:0000:0800:: e 3FFE:2B00:0100:0107::0001.
Deve-se ter o cuidado de suprimir somente uma seqüência de 16 bits. Ou seja, considerando o exemplo
1.2.2, ele não poderia ficar assim 1080:0008::0800::. Pode se ver que ele tem duas vezes o sinal de dois
pontos juntos o que seria inválido. Somente se pode ter uma única vez os sinais de dois pontos juntos [HID
98].

página 17 - Segundo e terceiro paragrafo do item 2.2"""
    },
    {
        "question": "Qual foi a principal motivação para o desenvolvimento do protocolo IPv6, considerando o contexto de esgotamento dos endereços IPv4 na década de 1990?",
        "answer": """Desde 1990 já se sabia da necessidade de aumentar o endereçamento dos números IPs, devido ao grande
aumento que estava ocorrendo na Internet e que em pouco tempo se esgotariam os números IPs. Este
problema eminente foi resolvido em duas etapas, a primeira foi a redistribuição dos endereços através da
proposta do Classless Inter-Domain Routing (CIDR), especificado na RFC 1518 e 1519. Isto deu um tempo
a mais para a criação de uma nova proposta de endereçamento [SAL 00]. A segunda etapa foi a criação de
um novo protocolo que aumentaria os endereços IPs. Então a IETF (Internet Engineering Task Force) criou
um grupo de trabalho para desenvolver o novo protocolo chamado inicialmente de IPng (Internet Protocol
New Generation), o qual iria substituir o protocolo atual que passou a ser denominado IPv4 (Internet
Protocol version 4). Então, em 1994, após várias discussões e revisões de propostas para o novo protocolo,
o grupo de trabalho do IPng decidiu que caminho seguir. A proposta escolhida foi o SIPP (Simple IP Plus)
[SIL 97] - Página 16 - quarto paragrafo."""
    },
    {
        "question": "Quais campos do cabeçalho IPv4 foram eliminados ou modificados no cabeçalho base do IPv6 para simplificar o processamento pelos roteadores?",
        "answer": """Portanto, dos quatorze campos que existem na
versão anterior, passou-se a ter oito campos na nova versão do IP. Somente três campos continuam com a
mesma denominação e função, que são: version, source address e destination address. Alguns campos
como, por exemplo, o total length do IPv4 foi substituído por payload length e teve uma pequena mudança
como será visto mais adiante - página 26 - primeiro paragrafo."""
    },
    {
        "question": "Quais são as regras fundamentais para a implementação e uso correto do campo Flow Label no cabeçalho IPv6?",
        "answer": """Existem três regras a serem cumpridas quando é implementado o uso do flow label, estas regras são [STA
00]:

 - Host e roteadores que não suportam o flow label devem, ao criar o pacote, atribuir o valor zero para
o campo, não modificar o campo quando for repassar o pacote e ignorar o campo quando receber o
pacote.
 - Os pacotes criados por um nó com o mesmo valor e não zero do flow label devem ter o mesmo
endereço destino e origem, e se for o caso, o Hop-by-hop Options Header e Routing Header iguais
(estes dois cabeçalhos serão apresentados no capitulo dois).
 - Para obter um valor flow label para um fluxo de dados, deve ser escolhido um valor na faixa entre 1
e (220 – 1) e não se deve reusar o mesmo valor do flow label para um novo fluxo de dados quando
terminar o tempo de vida do primeiro. O valor zero é reservado para informar que o campo não está
sendo usado. Página 26 e 27 do item 2.4 - terceiro paragrafo."""
    },
    {
        "question": "Como o campo Payload Length do IPv6 difere do campo Total Length do IPv4 em termos de escopo de contagem?",
        "answer": """O campo Payload Length é equivalente ao campo total length (tamanho total) no IPv4. A diferença entre
eles é que, por definição, o payload length significa somente o tamanho de dados em octetos após o
cabeçalho, excluindo o tamanho do cabeçalho. Isso não ocorria no IPv4, que contabilizava tudo, inclusive o
cabeçalho [HUI 00]. 4 paragrafo - pagina 27"""
    },
    {
        "question": "Qual é a função do campo Hop Limit no IPv6 e como ele evita que pacotes fiquem indefinidamente em loop na rede?",
        "answer": """O campo Hop Limit é um inteiro positivo de 8 bits de tamanho. Sua funcionalidade é de saber quando o
pacote deve ser descartado por um roteador, assim, o pacote não ficará vagando eternamente pela rede, se
não for encontrado o seu destino. Funciona da seguinte forma: cada vez que o pacote é passado adiante
por um roteador, e seu valor é decrementado de um. Quando o valor for igual a zero, o pacote é descartado
[SAL 00]. sexto paragrafo - pagina 27"""
    },
    {
        "question": "Qual é o tamanho dos campos de endereço de origem e destino no IPv6 e qual a principal consequência desse aumento em relação à capacidade de endereçamento da Internet?",
        "answer": """Os campos source address e destination address são, respectivamente, os endereços de origem e destino
do pacote. Cada um destes campos possui o tamanho de 128 bits [DEE 98]. sétimo paragrafo - pagina 27"""
    },
    {
        "question": "O que caracteriza um endereço anycast e em que situações práticas ele é mais vantajoso que o unicast ou multicast?",
        "answer": """O conceito de endereço anycast é mais recente do que o unicast e o multicast. Pode-se dizer que o anycast
é uma mistura do unicast e do multicast. A finalidade deste endereço pode ser exemplificada numa situação,
onde se deseja requisitar um servidor de nomes de uma rede. A máquina requisitante irá enviar um pacote
para um determinado endereço anycast, que todos os servidores de nome possuem. O roteador se
encarregará de entregar o pacote ao servidor mais próximo da máquina que requisitou [HUI 98]. pagina 27 - primeiro paragrafo do item 2.3.3"""
    },
    {
        "question": "Quais são os dois tipos de endereços IPv6 utilizados para compatibilidade com redes IPv4?",
        "answer": """Endereço de translação para IPV4:
Como existe a idéia do IPv6 substituir aos poucos as redes IPv4, foi criado um mecanismo para fazer o
tunelamento dinamicamente de pacotes IPv6 para pacotes IPv4. Os nós que usam esta técnica têm um
endereço unicast especial, que carrega nos 32-bits de baixa prioridade um endereço IPv4. Os restantes dos
96 bits são representados por zeros. Esta técnica é chamada de IPv4, compatível com endereço IPv6 [HID
98]. O endereço 0:0:0:0:0:0:10.16.169.1 (ou ::10.16.169.1) é um exemplo dessa técnica.
Existe também um segundo tipo de endereço de translação que é o mapeamento do IPv4 para endereço
IPv6. Ou seja, serve para representar nós que não suportam endereços IPv6. Foi definido que este tipo de
endereço possui os 80 primeiros bits de alta prioridade com valores zero. Os 16 bits após têm o valor um e
os 32 bits restantes representam um endereço IPv4 [HID 98]. Um exemplo deste tipo de mapeamento é
representado pelo endereço 0:0:0:0:0:FFFF:10.16.169.1, ou simplificando ::FFFF:10.16.169.1. - página 20 - segundo paragrafo."""
    }
]

# Preparar o corpus de respostas
answers = [qa["answer"] for qa in qa_pairs]

print(f"Total de perguntas/respostas carregadas: {len(answers)}")

# Codificar todas as respostas para busca semântica
print("\nCodificando respostas para busca semântica...")
corpus_embeddings = bi_encoder.encode(answers, convert_to_tensor=True, show_progress_bar=True)

def search(query, expected_answer_index=None):
    """
    Função de busca que encontra respostas relevantes para uma pergunta
    """
    print(f"\n{'='*60}")
    print(f"Pergunta: {query}")
    print(f"{'='*60}")
    
    # Se temos o índice da resposta esperada, mostrar ela primeiro
    if expected_answer_index is not None:
        print("\n✅ Resposta correta esperada (primeiro parágrafo):")
        preview = answers[expected_answer_index][:200] + "..."
        print(f"   {preview}")
    
    ##### Busca Semântica #####
    # Codificar a pergunta usando o bi-encoder
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    if torch.cuda.is_available():
        question_embedding = question_embedding.cuda()
    
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]
    
    # Adicionar scores da busca semântica inicial
    for hit in hits:
        hit['retrieval-score'] = hit['score']
    
    ##### Re-Ranking com Cross-Encoder #####
    cross_inp = [[query, answers[hit["corpus_id"]]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)
    
    for idx in range(len(cross_scores)):
        hits[idx]["cross-score"] = cross_scores[idx]
    
    # Resultados da busca semântica inicial
    print(f"\n📊 Top-{min(3, len(hits))} resultados da Busca Semântica Inicial:")
    hits_retrieval = sorted(hits, key=lambda x: x["retrieval-score"], reverse=True)
    for i, hit in enumerate(hits_retrieval[:3]):
        preview = answers[hit["corpus_id"]][:100].replace("\n", " ") + "..."
        print(f"   {i+1}. Score: {hit['retrieval-score']:.3f} | {preview}")
        if expected_answer_index is not None and hit["corpus_id"] == expected_answer_index:
            print("      ⭐ ESTA É A RESPOSTA CORRETA!")
    
    # Resultados após re-ranking
    print(f"\n🎯 Top-{min(3, len(hits))} resultados após Re-ranking (Cross-Encoder):")
    hits_reranked = sorted(hits, key=lambda x: x["cross-score"], reverse=True)
    for i, hit in enumerate(hits_reranked[:3]):
        preview = answers[hit["corpus_id"]][:100].replace("\n", " ") + "..."
        print(f"   {i+1}. Score: {hit['cross-score']:.3f} | {preview}")
        if expected_answer_index is not None and hit["corpus_id"] == expected_answer_index:
            print("      ⭐ ESTA É A RESPOSTA CORRETA!")
    
    # Verificar se a resposta correta está no top-3 após re-ranking
    if expected_answer_index is not None:
        top3_ids = [hit["corpus_id"] for hit in hits_reranked[:3]]
        if expected_answer_index in top3_ids:
            print(f"\n✅ SUCESSO: A resposta correta está no Top-3 após re-ranking!")
        else:
            print(f"\n❌ A resposta correta NÃO está no Top-3 após re-ranking.")
            print(f"   Posição da resposta correta: {[hit['corpus_id'] for hit in hits_reranked].index(expected_answer_index) + 1}ª")

# Selecionar 3 perguntas aleatórias
print("\n🎲 Selecionando 3 perguntas aleatórias do conjunto...")
selected_indices = random.sample(range(len(qa_pairs)), 3)
print(f"Índices selecionados: {selected_indices}")

# Executar busca para cada pergunta selecionada
for idx in selected_indices:
    question = qa_pairs[idx]["question"]
    search(question, expected_answer_index=idx)
    print("\n" + "="*60 + "\n")

# Opcional: Teste com uma pergunta personalizada
print("\n🔍 Teste com pergunta personalizada:")
search("Como funciona o anycast no IPv6?")




LEGACY
=======


# @title Setup Env
!pip install -q transformers datasets accelerate bitsandbytes sentence-transformers chromadb \
                langchain langchain-huggingface langchain-chroma langchain-community langchain-ollama \
                --upgrade

# 2-stage IR with `langchain_classic` compressor/reranker

After many days combing through the LangChain docs, outdated tutorials, and mutually contradicting LLMs, I figured why I was having so much trouble adding a second stage my base information retrieval pipeline.

In `langchain > 1.x` a lot has changed from the older versions, including the setup of a HF cross-encoder reranker. Some of the classes used in "the old way" now are part of `langchain_classic`, which I guess means they are deprecated until further notice?

Anyway, I still wanted to give it a shot at assembling the pipeline myself from the sources I found (and extensive help from LLMs), even though it's probably (definitely) no longer the recommended way to go with newer LangChain versions.

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

import torch
import hashlib
import os

# @title Load pretrained embedder model


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}.")
model_kwargs = {'device': device, # Use 'cuda:0' to specify a GPU, if multiple available
                # 'device_map': 'auto' # Use this to automatically spread the model across multiple GPUs, or
                #       to offload parts of the model from the GPU to the CPU if it's too big for VRAM. DON'T SET BOTH
                }

model_name = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder"
embedding_model = HuggingFaceEmbeddings(model_name=model_name,
                                        model_kwargs=model_kwargs)
# Change max length for longer chunks (BERT-style models support a max sequence len of 512)
embedding_model._client.max_seq_length = 512

# Load tokenizer separately to use `RecursiveCharacterTextSplitter.from_huggingface_tokenizer`
embedding_tokenizer = AutoTokenizer.from_pretrained(model_name)

embedding_model

# @title Instantiate and populate vector DB with chunked dummy documents


print("Starting Chroma...")
# Setup ChromaDB
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embedding_model, # Embeddings are calculated on the fly
    persist_directory="./chroma_langchain_db", # Save data locally, or use host=[...] to connect to an existing ChromaDB server
    # host="localhost",
)

# Download "database"
!curl -o glossary.md https://raw.githubusercontent.com/GabrielvanderSchmidt/SENAC-DLGlossary/refs/heads/main/glossary.md

print("Doc-in' n' chunkin'...")
documents = [file for file in os.listdir(".") if os.path.isfile(file)]
documents_text = []
documents_meta = []
i = 0
for doc in documents:
    with open(doc, "r") as file:
        documents_text.append(file.read())
        documents_meta.append({"document_id" : str(i),
                               "tags" : ["learning", "AI"],
                               "source" : doc})
        i = i+1

# Convert strings to Document objects
documents = [Document(page_content=text, metadata=meta) for text, meta in zip(documents_text, documents_meta)]

# Chunk docs
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=embedding_tokenizer, # Load tokenizer to calculate chunk length
    chunk_size=200, # Chunk size (tokens) -- must be smaller than embedder sequence length
    chunk_overlap=30, # Chunk overlap (tokens) so context isn't lost at the "edges" of chunks
    add_start_index=True, # (Try to) Track index in original document
)

all_splits = text_splitter.split_documents(documents)
print(f"Documents split in {len(all_splits)} chunks.")

# Ids can be generated beforehand, or created by the `add_documents` call
# Let's generate them with our hashing function
def generate_ids(chunks):
    text_chunk = [chunk.page_content for chunk in chunks]
    encoded_chunks = [chunk.encode("utf-8") for chunk in text_chunk]
    hashes = [hashlib.md5(chunk).hexdigest() for chunk in encoded_chunks]
    return hashes

print("Embedding and pushing to vector DB...")
ids = generate_ids(all_splits)
vector_store.add_documents(documents=all_splits, ids=ids) # Outputs added ids

# Check chunking output
vector_store.get(ids=ids[0])

# @title Load pretrained cross-encoder model


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}.")
model_kwargs = {'device': device, # Use 'cuda:0' to specify a GPU, if multiple available
                # 'device_map': 'auto' # Use this to automatically spread the model across multiple GPUs, or to offload
                #        parts of the model from the GPU to the CPU if it's too big for VRAM. DON'T SET BOTH
                }

# model_name = "PORTULAN/albertina-1b5-portuguese-ptbr-encoder" # Final model we will want to use
model_name =  "cross-encoder/ms-marco-MiniLM-L-6-v2" # Placeholder model for now
cross_encoder_model = HuggingFaceCrossEncoder(model_name=model_name,
                                              model_kwargs=model_kwargs)
cross_encoder_model

# @title Setup two-stage retrieval pipeline


# Set vector DB + embedding model as a retriever (1st stage)
base_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20} # Retrieve a massive candidate pool for subsequent reranking
) # In this stage, we maximize recall (minimize false negatives - important context left out)

# Initialize reranking object (2nd stage)
reranker = CrossEncoderReranker(
    model=cross_encoder_model,
    top_n=3 # Retain only the 3 most strictly relevant chunks
) # In this stage, we maximize precision (minimize false positives - trash let in)

# Construct the unified two-stage retrieval pipeline
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

for i, doc in enumerate(compression_retriever.invoke("Qual é o componente mais fundamental de redes neurais?")):
    print(f"Top {i+1} match:")
    print(doc.page_content)




### 2026-03-23

Compreensão e instalação docanno: Ferramenta de anotação de dados. A ideia é usar isso para ter uma forma de avaliar o nosso pipeline independente de chunking/splitting do texto, 
pois teremos posições absolutas em relação ao começo do documento (ao invés de pares pergunta-resposta do gold standard original).
1) Entendimento processo Recuperação em Duas Etapas. Segue Codigo:

=====
from sentence_transformers import SentenceTransformer, util, CrossEncoder

# -------------------------------
# 1. Data base
# -------------------------------
qa_pairs = [
    {
        "question": "Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast?",
        "answer": """O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem parte
de um determinado endereço multicast. A vantagem em relação ao unicast é a economia de banda da rede,
pois transmite um único pacote para vários nós ao mesmo tempo. Algumas transmissões de vídeos usam multicast."""
    },
    {
        "question": "Além da forma preferencial de representação dos endereços IPv6, quais são os métodos de abreviação permitidos?",
        "answer": """Pode-se suprimir zeros à esquerda de cada bloco de 16 bits. Outra forma é substituir sequências de blocos
com valor zero por '::', mas isso só pode ser feito uma única vez no endereço."""
    },
    {
        "question": "Qual foi a principal motivação para o desenvolvimento do protocolo IPv6?",
        "answer": """O esgotamento dos endereços IPv4 devido ao crescimento da internet levou à criação do IPv6, com maior capacidade de endereçamento."""
    },
    {
        "question": "Quais campos do cabeçalho IPv4 foram eliminados ou modificados no IPv6?",
        "answer": """O IPv6 reduziu o número de campos no cabeçalho, mantendo alguns como version e endereços, e substituindo outros como total length por payload length."""
    },
    {
        "question": "Quais são as regras do campo Flow Label no IPv6?",
        "answer": """Hosts que não suportam devem usar valor zero. Pacotes com mesmo flow label devem ter mesmos endereços de origem e destino.
O valor deve estar entre 1 e (2^20 - 1) e não deve ser reutilizado rapidamente."""
    },
    {
        "question": "Como o campo Payload Length difere do Total Length?",
        "answer": """O Payload Length conta apenas os dados após o cabeçalho, enquanto o Total Length do IPv4 inclui o cabeçalho."""
    },
    {
        "question": "Qual é a função do campo Hop Limit no IPv6?",
        "answer": """O Hop Limit evita que pacotes fiquem em loop. Ele é decrementado a cada roteador e quando chega a zero o pacote é descartado."""
    },
    {
        "question": "Qual é o tamanho dos endereços IPv6?",
        "answer": """Os endereços IPv6 possuem 128 bits, permitindo uma quantidade muito maior de endereços na internet."""
    },
    {
        "question": "O que é um endereço anycast?",
        "answer": """Anycast permite enviar um pacote para o servidor mais próximo dentro de um grupo, sendo útil para serviços distribuídos."""
    },
    {
        "question": "Quais são os tipos de compatibilidade IPv4 no IPv6?",
        "answer": """Existem endereços IPv4-compatíveis (::IPv4) e IPv4-mapeados (::FFFF:IPv4), usados para integração entre redes."""
    }
]

# -------------------------------
# 2. Pergunta do usuário
# -------------------------------
pergunta_usuario = "Para que serve o Hop Limit no IPv6?"

# -------------------------------
# 3. ETAPA 1 → Embedding (rápido)
# -------------------------------
modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')

# PERGUNTA + RESPOSTA juntas
textos = [qa["question"] + " " + qa["answer"] for qa in qa_pairs]

emb_textos = modelo_embedding.encode(textos, convert_to_tensor=True)
emb_pergunta = modelo_embedding.encode(pergunta_usuario, convert_to_tensor=True)

similaridades = util.cos_sim(emb_pergunta, emb_textos)[0]

top_k = 5
top_resultados = similaridades.topk(k=top_k)

candidatos = [qa_pairs[i] for i in top_resultados.indices]

print("\n🔎 Etapa 1 - Candidatos:")
for c in candidatos:
    print("-", c["question"])

# -------------------------------
# 4. ETAPA 2 → Cross Encoder (preciso)
# -------------------------------
modelo_cross = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pares = [
    [pergunta_usuario, c["question"] + " " + c["answer"]]
    for c in candidatos
]

scores = modelo_cross.predict(pares)

# Ordena
resultados_finais = sorted(
    zip(candidatos, scores),
    key=lambda x: x[1],
    reverse=True
)

# -------------------------------
# 5. Resultado final
# -------------------------------
print("\n🎯 Melhor resposta:\n")

melhor = resultados_finais[0]

print(f"Score: {melhor[1]:.4f}")
print("\nPergunta relacionada:")
print(melhor[0]["question"])

print("\nResposta:")
print(melhor[0]["answer"])


----------
2) Avaliar se está encontrando os trechos corretos. Segue código:
# ================================
# Instalação de bibliotecas
# ================================
!pip install -qU langchain sentence-transformers faiss-cpu langchain_text_splitters langchain_community

# ================================
# 1. Imports
# ================================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# ================================
# 2. dataset original
# ================================
qa_pairs = [
    {
        "question": "Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast?",
        "answer": """O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem parte
de um determinado endereço multicast. A vantagem em relação ao unicast é a economia de banda da rede,
pois transmite um único pacote para vários nós ao mesmo tempo. Algumas transmissões de vídeos usam multicast."""
    },
    {
        "question": "Além da forma preferencial de representação dos endereços IPv6, quais são os métodos de abreviação permitidos?",
        "answer": """Pode-se suprimir zeros à esquerda de cada bloco de 16 bits. Outra forma é substituir sequências de blocos
com valor zero por '::', mas isso só pode ser feito uma única vez no endereço."""
    },
    {
        "question": "Qual foi a principal motivação para o desenvolvimento do protocolo IPv6?",
        "answer": """O esgotamento dos endereços IPv4 devido ao crescimento da internet levou à criação do IPv6, com maior capacidade de endereçamento."""
    },
    {
        "question": "Quais campos do cabeçalho IPv4 foram eliminados ou modificados no IPv6?",
        "answer": """O IPv6 reduziu o número de campos no cabeçalho, mantendo alguns como version e endereços, e substituindo outros como total length por payload length."""
    },
    {
        "question": "Quais são as regras do campo Flow Label no IPv6?",
        "answer": """Hosts que não suportam devem usar valor zero. Pacotes com mesmo flow label devem ter mesmos endereços de origem e destino.
O valor deve estar entre 1 e (2^20 - 1) e não deve ser reutilizado rapidamente."""
    },
    {
        "question": "Como o campo Payload Length difere do Total Length?",
        "answer": """O Payload Length conta apenas os dados após o cabeçalho, enquanto o Total Length do IPv4 inclui o cabeçalho."""
    },
    {
        "question": "Qual é a função do campo Hop Limit no IPv6?",
        "answer": """O Hop Limit evita que pacotes fiquem em loop. Ele é decrementado a cada roteador e quando chega a zero o pacote é descartado."""
    },
    {
        "question": "Qual é o tamanho dos endereços IPv6?",
        "answer": """Os endereços IPv6 possuem 128 bits, permitindo uma quantidade muito maior de endereços na internet."""
    },
    {
        "question": "O que é um endereço anycast?",
        "answer": """Anycast permite enviar um pacote para o servidor mais próximo dentro de um grupo, sendo útil para serviços distribuídos."""
    },
    {
        "question": "Quais são os tipos de compatibilidade IPv4 no IPv6?",
        "answer": """Existem endereços IPv4-compatíveis (::IPv4) e IPv4-mapeados (::FFFF:IPv4), usados para integração entre redes."""
    }
]

# ================================
# 3. Criar documento único (base)
# ================================
documento = "\n\n".join([qa["answer"] for qa in qa_pairs])

# ================================
# 4. Gerar posições automaticamente
# ================================
cursor = 0

for qa in qa_pairs:
    start = documento.find(qa["answer"], cursor)
    end = start + len(qa["answer"])

    qa["answer_start"] = start
    qa["answer_end"] = end

    cursor = end  # evita conflitos

# ================================
# 5. Chunking com posição
# ================================
def chunk_text_with_position(texto, chunk_size=300, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = []
    start = 0

    for chunk in splitter.split_text(texto):
        idx = texto.find(chunk, start)

        chunks.append({
            "text": chunk,
            "start": idx,
            "end": idx + len(chunk)
        })

        start = idx + 1

    return chunks

chunks = chunk_text_with_position(documento)

# ================================
# 6. Criar documentos LangChain
# ================================
docs = [
    Document(
        page_content=c["text"],
        metadata={"start": c["start"], "end": c["end"]}
    )
    for c in chunks
]

# ================================
# 7. Embeddings LOCAIS + FAISS (SEM API KEY!)
# ================================
print("🔄 Carregando embeddings locais (sentence-transformers)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("✅ Vectorstore criado com sucesso!")

# ================================
# 8. Função de overlap
# ================================
def calcular_overlap(chunk_start, chunk_end, ans_start, ans_end):
    inter_inicio = max(chunk_start, ans_start)
    inter_fim = min(chunk_end, ans_end)

    inter = max(0, inter_fim - inter_inicio)
    tamanho_resposta = ans_end - ans_start

    if inter == 0:
        return 0

    return inter / tamanho_resposta

# ================================
# 9. Avaliação
# ================================
def avaliar(qa_pairs, retriever):
    total_corretos = 0
    total_encontrados = 0
    total_relevantes = len(qa_pairs)

    resultados = []

    for qa in qa_pairs:
        docs_encontrados = retriever.invoke(qa["question"])

        encontrou_completo = False

        for doc in docs_encontrados:
            score = calcular_overlap(
                doc.metadata["start"],
                doc.metadata["end"],
                qa["answer_start"],
                qa["answer_end"]
            )

            if score == 1:
                encontrou_completo = True

        if encontrou_completo:
            total_corretos += 1

        total_encontrados += len(docs_encontrados)

        resultados.append({
            "pergunta": qa["question"],
            "acerto": encontrou_completo
        })

    # ================================
    # 10. Métricas
    # ================================
    precision_at_k = total_corretos / total_encontrados if total_encontrados > 0 else 0
    recall_at_k = total_corretos / total_relevantes
    accuracy = total_corretos / total_relevantes

    return resultados, {
        "Precision@K": precision_at_k,
        "Recall@K": recall_at_k,
        "Accuracy": accuracy
    }

# ================================
# 11. Executar
# ================================
print("🚀 Executando avaliação...")
resultados, metricas = avaliar(qa_pairs, retriever)

print("\n" + "="*80)
print("📋 RESULTADOS DETALHADOS")
print("="*80)

for r in resultados:
    status = "✅ ACERTOU" if r["acerto"] else "❌ ERROU"
    print(f"\n{status}")
    print(f"Pergunta: {r['pergunta'][:100]}...")

print("\n" + "="*80)
print("📊 MÉTRICAS FINAIS")
print("="*80)
for k, v in metricas.items():
    print(f"🎯 {k}: {v:.3f}")

print(f"\n📈 Resumo: {len([r for r in resultados if r['acerto']])}/{len(resultados)} perguntas respondidas corretamente")


### 2026-03-25

1)Pesquisas e experimentação para:
a)Avaliar o  pipeline de forma completa;
b) Avaliar partes (chunking / retrieval)
c)Criar as propiras métricas
d) Trabalhar sem LangSmith
=========================================================================================================================================
Codigo(1): Vai fazer o seguinte: 
(i) Usar seus qa_pairs como gold standard;
(ii) Simular retrieval simples;
(iii) Gerar respostas;
(iv) Avaliar com as métricas: a - Similaridade (resposta vs gold); Retrieval Score (chunk certo foi recuperado?)

from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =========================
# 1. (GOLD)
# =========================
qa_pairs = [
    {
        "question": "Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast?",
        "answer": """O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem parte
de um determinado endereço multicast. A vantagem em relação ao unicast é a economia de banda da rede,
pois transmite um único pacote para vários nós ao mesmo tempo. Algumas transmissões de vídeos usam multicast."""
    },
    {
        "question": "Além da forma preferencial de representação dos endereços IPv6, quais são os métodos de abreviação permitidos?",
        "answer": """Pode-se suprimir zeros à esquerda de cada bloco de 16 bits. Outra forma é substituir sequências de blocos
com valor zero por '::', mas isso só pode ser feito uma única vez no endereço."""
    },
    {
        "question": "Qual foi a principal motivação para o desenvolvimento do protocolo IPv6?",
        "answer": """O esgotamento dos endereços IPv4 devido ao crescimento da internet levou à criação do IPv6, com maior capacidade de endereçamento."""
    },
    {
        "question": "Quais campos do cabeçalho IPv4 foram eliminados ou modificados no IPv6?",
        "answer": """O IPv6 reduziu o número de campos no cabeçalho, mantendo alguns como version e endereços, e substituindo outros como total length por payload length."""
    },
    {
        "question": "Quais são as regras do campo Flow Label no IPv6?",
        "answer": """Hosts que não suportam devem usar valor zero. Pacotes com mesmo flow label devem ter mesmos endereços de origem e destino.
O valor deve estar entre 1 e (2^20 - 1) e não deve ser reutilizado rapidamente."""
    },
    {
        "question": "Como o campo Payload Length difere do Total Length?",
        "answer": """O Payload Length conta apenas os dados após o cabeçalho, enquanto o Total Length do IPv4 inclui o cabeçalho."""
    },
    {
        "question": "Qual é a função do campo Hop Limit no IPv6?",
        "answer": """O Hop Limit evita que pacotes fiquem em loop. Ele é decrementado a cada roteador e quando chega a zero o pacote é descartado."""
    },
    {
        "question": "Qual é o tamanho dos endereços IPv6?",
        "answer": """Os endereços IPv6 possuem 128 bits, permitindo uma quantidade muito maior de endereços na internet."""
    },
    {
        "question": "O que é um endereço anycast?",
        "answer": """Anycast permite enviar um pacote para o servidor mais próximo dentro de um grupo, sendo útil para serviços distribuídos."""
    },
    {
        "question": "Quais são os tipos de compatibilidade IPv4 no IPv6?",
        "answer": """Existem endereços IPv4-compatíveis (::IPv4) e IPv4-mapeados (::FFFF:IPv4), usados para integração entre redes."""
    }
]

# =========================
# 2. MODELO DE EMBEDDING
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# 3. INDEXAÇÃO (SIMULA RAG)
# =========================
documents = [item["answer"] for item in qa_pairs]
doc_embeddings = model.encode(documents)

# =========================
# 4. FUNÇÃO DE RETRIEVAL
# =========================
def retrieve_context(question: str, top_k=1):
    q_embedding = model.encode([question])
    similarities = cosine_similarity(q_embedding, doc_embeddings)[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = [(documents[i], similarities[i]) for i in top_indices]
    return results

# =========================
# 5. "GERAÇÃO" (SIMPLES)
# =========================
def generate_answer(context: str):
    

# =========================
# 6. MÉTRICAS
# =========================
def similarity_score(a: str, b: str):
    emb_a = model.encode([a])
    emb_b = model.encode([b])
    return cosine_similarity(emb_a, emb_b)[0][0]

# =========================
# 7. AVALIAÇÃO COMPLETA
# =========================
def evaluate_pipeline(qa_pairs):
    results = []
    
    for item in qa_pairs:
        question = item["question"]
        gold = item["answer"]
        
        retrieved = retrieve_context(question, top_k=1)
        context, score = retrieved[0]
        
        generated = generate_answer(context)
        
        sim = similarity_score(generated, gold)
        
        results.append({
            "question": question,
            "retrieval_score": score,
            "answer_similarity": sim
        })
    
    return results

# =========================
# 8. RODAR AVALIAÇÃO
# =========================
results = evaluate_pipeline(qa_pairs)

# =========================
# 9. PRINT RESULTADOS
# =========================
for r in results:
    print("Pergunta:", r["question"])
    print("Score de Recuperação:", round(r["retrieval_score"], 3))
    print("Similaridade da Resposta:", round(r["answer_similarity"], 3))
    print("-" * 50)

=========================================================================================================================

Código (2):
Aperfeiçoado com
Chunking Simples;
IoU de Texto;
LLM Real (Ollama);
Dataset 10x Maior;
3 Métricas Avaliadas: Retrieval Score (cosine similarity)
Similaridade Semântica (embeddings)
IoU de Palavras (overlap textual)

# ================================
# Instalação de bibliotecas
# ================================
!pip install -qU sentence-transformers faiss-cpu langchain_text_splitters langchain_community ollama

# ================================
# 1. Imports
# ================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import ollama
import re

# ================================
# 2. SEU DATASET ORIGINAL (GOLD)
# ================================
qa_pairs = [
    {
        "question": "Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast?",
        "answer": """O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem parte
de um determinado endereço multicast. A vantagem em relação ao unicast é a economia de banda da rede,
pois transmite um único pacote para vários nós ao mesmo tempo. Algumas transmissões de vídeos usam multicast."""
    },
    {
        "question": "Além da forma preferencial de representação dos endereços IPv6, quais são os métodos de abreviação permitidos?",
        "answer": """Pode-se suprimir zeros à esquerda de cada bloco de 16 bits. Outra forma é substituir sequências de blocos
com valor zero por '::', mas isso só pode ser feito uma única vez no endereço."""
    },
    {
        "question": "Qual foi a principal motivação para o desenvolvimento do protocolo IPv6?",
        "answer": """O esgotamento dos endereços IPv4 devido ao crescimento da internet levou à criação do IPv6, com maior capacidade de endereçamento."""
    },
    {
        "question": "Quais campos do cabeçalho IPv4 foram eliminados ou modificados no IPv6?",
        "answer": """O IPv6 reduziu o número de campos no cabeçalho, mantendo alguns como version e endereços, e substituindo outros como total length por payload length."""
    },
    {
        "question": "Quais são as regras do campo Flow Label no IPv6?",
        "answer": """Hosts que não suportam devem usar valor zero. Pacotes com mesmo flow label devem ter mesmos endereços de origem e destino.
O valor deve estar entre 1 e (2^20 - 1) e não deve ser reutilizado rapidamente."""
    },
    {
        "question": "Como o campo Payload Length difere do Total Length?",
        "answer": """O Payload Length conta apenas os dados após o cabeçalho, enquanto o Total Length do IPv4 inclui o cabeçalho."""
    },
    {
        "question": "Qual é a função do campo Hop Limit no IPv6?",
        "answer": """O Hop Limit evita que pacotes fiquem em loop. Ele é decrementado a cada roteador e quando chega a zero o pacote é descartado."""
    },
    {
        "question": "Qual é o tamanho dos endereços IPv6?",
        "answer": """Os endereços IPv6 possuem 128 bits, permitindo uma quantidade muito maior de endereços na internet."""
    },
    {
        "question": "O que é um endereço anycast?",
        "answer": """Anycast permite enviar um pacote para o servidor mais próximo dentro de um grupo, sendo útil para serviços distribuídos."""
    },
    {
        "question": "Quais são os tipos de compatibilidade IPv4 no IPv6?",
        "answer": """Existem endereços IPv4-compatíveis (::IPv4) e IPv4-mapeados (::FFFF:IPv4), usados para integração entre redes."""
    }
]

# ================================
# 3. CHUNKING SIMPLES (REQUISITO 1)
# ================================
def chunk_text(text, size=100):
    """Divide texto em chunks de tamanho fixo"""
    return [text[i:i+size] for i in range(0, len(text), size)]

# ================================
# 4. IoU DE TEXTO (REQUISITO 2)
# ================================
def iou_text(a: str, b: str):
    """IoU baseado em palavras únicas"""
    set_a = set(a.split())
    set_b = set(b.split())
    
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return inter / union if union > 0 else 0

# ================================
# 5. MODELO DE EMBEDDING
# ================================
print("🔄 Carregando modelo de embedding...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ================================
# 6. GERADOR DE PERGUNTAS COM LLM (REQUISITO 4)
# ================================
def generate_questions(answer: str, num_questions: int = 5) -> List[str]:
    """Gera 5 perguntas por resposta usando LLM local"""
    prompt = f"""Com base nesta resposta técnica sobre IPv6, gere EXATAMENTE {num_questions} perguntas que teriam esta resposta:

Resposta: {answer}

Perguntas (numeradas 1-{num_questions}):"""
    
    try:
        response = ollama.generate(
            model='llama3.2:1b',  # Modelo leve e rápido
            prompt=prompt,
            options={'num_predict': 300}
        )
        generated_text = response['response']
        
        # Extrair perguntas numeradas
        questions = []
        lines = generated_text.split('\n')
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                questions.append(line.strip())
                if len(questions) >= num_questions:
                    break
        
        return questions[:num_questions]
    except:
        # Fallback se Ollama não estiver disponível
        return [f"Pergunta {i} sobre: {answer[:50]}..." for i in range(num_questions)]

# ================================
# 7. EXPANDIR DATASET 10x (REQUISITO 4)
# ================================
print("🔄 Gerando dataset expandido (10x)...")
expanded_qa_pairs = []

for qa in qa_pairs:
    # Adiciona pergunta original
    expanded_qa_pairs.append(qa)
    
    # Gera 5 perguntas adicionais
    new_questions = generate_questions(qa["answer"])
    for q in new_questions:
        expanded_qa_pairs.append({
            "question": q,
            "answer": qa["answer"]
        })

print(f"✅ Dataset expandido: {len(expanded_qa_pairs)} perguntas (original: {len(qa_pairs)})")

# ================================
# 8. LLM REAL PARA GERAÇÃO (REQUISITO 3)
# ================================
def generate_answer(question: str, context: str) -> str:
    """Gera resposta usando LLM real com RAG"""
    prompt = f"""Pergunta: {question}

Contexto: {context}

Responda APENAS com base no contexto fornecido, de forma concisa e precisa:"""
    
    try:
        response = ollama.generate(
            model='llama3.2:1b',
            prompt=prompt,
            options={'num_predict': 200}
        )
        return response['response'].strip()
    except:
        # Fallback
        return context[:200] + "..."

# ================================
# 9. PROCESSAMENTO DE CHUNKS E INDEXAÇÃO
# ================================
print("🔄 Processando chunks e criando índice...")
all_chunks = []
chunk_to_gold_map = {}

for i, qa in enumerate(qa_pairs):  # Só usa gold answers para chunks
    chunks = chunk_text(qa["answer"], size=100)
    all_chunks.extend(chunks)
    for chunk in chunks:
        chunk_to_gold_map[chunk] = qa["answer"]

# Embedding dos chunks
chunk_embeddings = model.encode(all_chunks)
print(f"✅ {len(all_chunks)} chunks indexados")

# ================================
# 10. RETRIEVAL MELHORADO
# ================================
def retrieve_context(question: str, top_k=3):
    """Recupera chunks mais relevantes"""
    q_embedding = model.encode([question])
    similarities = cosine_similarity(q_embedding, chunk_embeddings)[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    retrieved_chunks = [all_chunks[i] for i in top_indices]
    context = " ".join(retrieved_chunks)
    
    return context, similarities[top_indices]

# ================================
# 11. AVALIAÇÃO COMPLETA
# ================================
def evaluate_pipeline(test_pairs: List[Dict]):
    """Avalia pipeline completo com todas as métricas"""
    results = []
    
    for item in test_pairs:
        question = item["question"]
        gold_answer = item["answer"]
        
        # Retrieval
        context, retrieval_scores = retrieve_context(question, top_k=3)
        retrieval_score = retrieval_scores[0]  # Melhor score
        
        # Geração com LLM
        generated_answer = generate_answer(question, context)
        
        # Métricas
        semantic_sim = cosine_similarity(
            model.encode([generated_answer]),
            model.encode([gold_answer])
        )[0][0]
        
        iou_score = iou_text(generated_answer, gold_answer)
        
        results.append({
            "question": question[:100] + "..." if len(question) > 100 else question,
            "retrieval_score": retrieval_score,
            "semantic_similarity": semantic_sim,
            "iou_score": iou_score,
            "gold_answer": gold_answer[:100] + "...",
            "generated_answer": generated_answer[:100] + "..."
        })
    
    return results

# ================================
# 12. EXECUTAR AVALIAÇÃO
# ================================
print("\n🚀 Executando avaliação completa...")
results = evaluate_pipeline(expanded_qa_pairs[:20])  # Testa 20 primeiras para velocidade

# ================================
# 13. MÉTRICAS AGREGADAS
# ================================
avg_retrieval = np.mean([r["retrieval_score"] for r in results])
avg_semantic = np.mean([r["semantic_similarity"] for r in results])
avg_iou = np.mean([r["iou_score"] for r in results])

print("\n" + "="*80)
print("📊 RESULTADOS DA AVALIAÇÃO")
print("="*80)
print(f"🎯 Média Retrieval Score: {avg_retrieval:.3f}")
print(f"🎯 Média Similaridade Semântica: {avg_semantic:.3f}")
print(f"🎯 Média IoU (Texto): {avg_iou:.3f}")
print(f"📈 Total de perguntas testadas: {len(results)}")
print("="*80)

# ================================
# 14. RESULTADOS DETALHADOS (5 primeiros)
# ================================
print("\n📋 5 PRIMEIROS RESULTADOS:")
print("-" * 80)
for i, r in enumerate(results[:5]):
    print(f"\n{i+1}. Pergunta: {r['question']}")
    print(f"   Retrieval: {r['retrieval_score']:.3f}")
    print(f"   Semântica: {r['semantic_similarity']:.3f}")
    print(f"   IoU: {r['iou_score']:.3f}")
    print(f"   Gold: {r['gold_answer']}")
    print(f"   Gerado: {r['generated_answer']}")

==================================================================================================================================================================

### 2026-26-03 e 2006-27-03

Adaptaçao codigo docanno.

Gerado arquivo json para localizacao das respostas no PDF.

import json
import re
from pathlib import Path


# Lista de QA pairs 
qa_pairs = [
    {
        "question": "Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast?",
        "answer": """O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem parte
de um determinado endereço multicast. A vantagem que uma transmissão tem em relação ao unicast é a
economia de banda da rede, pois transmite um único pacote para vários nós ao mesmo tempo, o que não
ocorre com uma transmissão unicast, onde é transmitido um pacote para cada nó da rede. Algumas
transmissões de vídeos pela rede usam multicast, economizando assim a banda da rede - Pagina 22 - Primeiro paragrafo do item 2.3.2"""
    },
    {
        "question": "Além da forma preferencial de representação dos endereços IPv6, quais são os métodos de abreviação permitidos?",
        "answer": """i) Um dos métodos utilizados na representação do endereçamento é de suprimir os zeros da esquerda de cada
parte de 16 bits [HID 98]. Para exemplificar este método, usaremos como exemplo “1.2.2”. Aplicando o
método de representação, o exemplo ficaria da seguinte forma:
1080:8:0:800:0:0:0:0.;

ii) Outra forma de simplificar é suprimir as seqüências de 16 bits com valores iguais a zero. Assim, os
exemplos 1.2.2 e 1.2.3 ficariam respectivamente 1080:0008:0000:0800:: e 3FFE:2B00:0100:0107::0001.
Deve-se ter o cuidado de suprimir somente uma seqüência de 16 bits. Ou seja, considerando o exemplo
1.2.2, ele não poderia ficar assim 1080:0008::0800::. Pode se ver que ele tem duas vezes o sinal de dois
pontos juntos o que seria inválido. Somente se pode ter uma única vez os sinais de dois pontos juntos [HID
98].

página 17 - Segundo e terceiro paragrafo do item 2.2"""
    },
    {
        "question": "Qual foi a principal motivação para o desenvolvimento do protocolo IPv6, considerando o contexto de esgotamento dos endereços IPv4 na década de 1990?",
        "answer": """Desde 1990 já se sabia da necessidade de aumentar o endereçamento dos números IPs, devido ao grande
aumento que estava ocorrendo na Internet e que em pouco tempo se esgotariam os números IPs. Este
problema eminente foi resolvido em duas etapas, a primeira foi a redistribuição dos endereços através da
proposta do Classless Inter-Domain Routing (CIDR), especificado na RFC 1518 e 1519. Isto deu um tempo
a mais para a criação de uma nova proposta de endereçamento [SAL 00]. A segunda etapa foi a criação de
um novo protocolo que aumentaria os endereços IPs. Então a IETF (Internet Engineering Task Force) criou
um grupo de trabalho para desenvolver o novo protocolo chamado inicialmente de IPng (Internet Protocol
New Generation), o qual iria substituir o protocolo atual que passou a ser denominado IPv4 (Internet
Protocol version 4). Então, em 1994, após várias discussões e revisões de propostas para o novo protocolo,
o grupo de trabalho do IPng decidiu que caminho seguir. A proposta escolhida foi o SIPP (Simple IP Plus)
[SIL 97] - Página 16 - quarto paragrafo."""
    },
    {
        "question": "Quais campos do cabeçalho IPv4 foram eliminados ou modificados no cabeçalho base do IPv6 para simplificar o processamento pelos roteadores?",
        "answer": """Portanto, dos quatorze campos que existem na
versão anterior, passou-se a ter oito campos na nova versão do IP. Somente três campos continuam com a
mesma denominação e função, que são: version, source address e destination address. Alguns campos
como, por exemplo, o total length do IPv4 foi substituído por payload length e teve uma pequena mudança
como será visto mais adiante - página 26 - primeiro paragrafo."""
    },
    {
        "question": "Quais são as regras fundamentais para a implementação e uso correto do campo Flow Label no cabeçalho IPv6?",
        "answer": """Existem três regras a serem cumpridas quando é implementado o uso do flow label, estas regras são [STA
00]:

 - Host e roteadores que não suportam o flow label devem, ao criar o pacote, atribuir o valor zero para
o campo, não modificar o campo quando for repassar o pacote e ignorar o campo quando receber o
pacote.
 - Os pacotes criados por um nó com o mesmo valor e não zero do flow label devem ter o mesmo
endereço destino e origem, e se for o caso, o Hop-by-hop Options Header e Routing Header iguais
(estes dois cabeçalhos serão apresentados no capitulo dois).
 - Para obter um valor flow label para um fluxo de dados, deve ser escolhido um valor na faixa entre 1
e (220 – 1) e não se deve reusar o mesmo valor do flow label para um novo fluxo de dados quando
terminar o tempo de vida do primeiro. O valor zero é reservado para informar que o campo não está
sendo usado. Página 26 e 27 do item 2.4 - terceiro paragrafo."""
    },
    {
        "question": "Como o campo Payload Length do IPv6 difere do campo Total Length do IPv4 em termos de escopo de contagem?",
        "answer": """O campo Payload Length é equivalente ao campo total length (tamanho total) no IPv4. A diferença entre
eles é que, por definição, o payload length significa somente o tamanho de dados em octetos após o
cabeçalho, excluindo o tamanho do cabeçalho. Isso não ocorria no IPv4, que contabilizava tudo, inclusive o
cabeçalho [HUI 00]. 4 paragrafo - pagina 27"""
    },
    {
        "question": "Qual é a função do campo Hop Limit no IPv6 e como ele evita que pacotes fiquem indefinidamente em loop na rede?",
        "answer": """O campo Hop Limit é um inteiro positivo de 8 bits de tamanho. Sua funcionalidade é de saber quando o
pacote deve ser descartado por um roteador, assim, o pacote não ficará vagando eternamente pela rede, se
não for encontrado o seu destino. Funciona da seguinte forma: cada vez que o pacote é passado adiante
por um roteador, e seu valor é decrementado de um. Quando o valor for igual a zero, o pacote é descartado
[SAL 00]. sexto paragrafo - pagina 27"""
    },
    {
        "question": "Qual é o tamanho dos campos de endereço de origem e destino no IPv6 e qual a principal consequência desse aumento em relação à capacidade de endereçamento da Internet?",
        "answer": """Os campos source address e destination address são, respectivamente, os endereços de origem e destino
do pacote. Cada um destes campos possui o tamanho de 128 bits [DEE 98]. sétimo paragrafo - pagina 27"""
    },
    {
        "question": "O que caracteriza um endereço anycast e em que situações práticas ele é mais vantajoso que o unicast ou multicast?",
        "answer": """O conceito de endereço anycast é mais recente do que o unicast e o multicast. Pode-se dizer que o anycast
é uma mistura do unicast e do multicast. A finalidade deste endereço pode ser exemplificada numa situação,
onde se deseja requisitar um servidor de nomes de uma rede. A máquina requisitante irá enviar um pacote
para um determinado endereço anycast, que todos os servidores de nome possuem. O roteador se
encarregará de entregar o pacote ao servidor mais próximo da máquina que requisitou [HUI 98]. pagina 27 - primeiro paragrafo do item 2.3.3"""
    },
    {
        "question": "Quais são os dois tipos de endereços IPv6 utilizados para compatibilidade com redes IPv4?",
        "answer": """Endereço de translação para IPV4:
Como existe a idéia do IPv6 substituir aos poucos as redes IPv4, foi criado um mecanismo para fazer o
tunelamento dinamicamente de pacotes IPv6 para pacotes IPv4. Os nós que usam esta técnica têm um
endereço unicast especial, que carrega nos 32-bits de baixa prioridade um endereço IPv4. Os restantes dos
96 bits são representados por zeros. Esta técnica é chamada de IPv4, compatível com endereço IPv6 [HID
98]. O endereço 0:0:0:0:0:0:10.16.169.1 (ou ::10.16.169.1) é um exemplo dessa técnica.
Existe também um segundo tipo de endereço de translação que é o mapeamento do IPv4 para endereço
IPv6. Ou seja, serve para representar nós que não suportam endereços IPv6. Foi definido que este tipo de
endereço possui os 80 primeiros bits de alta prioridade com valores zero. Os 16 bits após têm o valor um e
os 32 bits restantes representam um endereço IPv4 [HID 98]. Um exemplo deste tipo de mapeamento é
representado pelo endereço 0:0:0:0:0:FFFF:10.16.169.1, ou simplificando ::FFFF:10.16.169.1. - página 20 - segundo paragrafo."""
    }
]

# Mapeamento de ordinal para índice zero-based
ordinal_to_index = {
    "primeiro": 0,
    "segundo": 1,
    "terceiro": 2,
    "quarto": 3,
    "quinto": 4,
    "sexto": 5,
    "sétimo": 6,
    "oitavo": 7,
    "nono": 8,
    "décimo": 9
}

def extract_page_and_paragraph(answer_text):
    """
    Extract page number and paragraph index from answer text.
    Returns (page, paragraph_index) or (None, None) if not found.
    """
    # Buscar página
    page_match = re.search(r'[Pp]agina\s+(\d+)', answer_text)
    if not page_match:
        return None, None
    page = int(page_match.group(1))

    # Buscar descrição do parágrafo
    # Expressão regular para capturar palavras como "primeiro", "segundo", "terceiro", "quarto", "quinto", "sexto", "sétimo", "oitavo", "nono", "décimo"
    para_match = re.search(r'\b(primeiro|segundo|terceiro|quarto|quinto|sexto|sétimo|oitavo|nono|décimo)\s+paragrafo', answer_text.lower())
    if para_match:
        ordinal = para_match.group(1)
        para_index = ordinal_to_index.get(ordinal, 0)
    else:
        # Se não encontrar ordinal, tenta algo como "4 paragrafo" (número)
        num_match = re.search(r'(\d+)\s+paragrafo', answer_text.lower())
        if num_match:
            para_index = int(num_match.group(1)) - 1  # 1-based to 0-based
        else:
            # fallback: assume primeiro parágrafo
            para_index = 0

    return page, para_index

# Gerar anotações no formato JSONL
annotations = []
for item in qa_pairs:
    question = item["question"]
    answer = item["answer"]
    page, para_idx = extract_page_and_paragraph(answer)
    if page is None:
        print(f"Aviso: não foi possível extrair página da resposta para: {question[:50]}...")
        continue
    annotations.append({
        "question": question,
        "page": page,
        "paragraph": para_idx
    })

# Caminho de saída (ajuste conforme desejar)
output_path = "/content/drive/MyDrive/UC15/annotations.jsonl"

# Salvar arquivo
with open(output_path, "w", encoding="utf-8") as f:
    for ann in annotations:
        f.write(json.dumps(ann, ensure_ascii=False) + "\n")

print(f"Arquivo de anotações salvo em: {output_path}")
print(f"Total de anotações: {len(annotations)}")


### 2026-03-30

Inicio script para fine tunning do Serafim - 100 mm.
a) Rodar no docker. Segue código:
import logging
import traceback
from datasets import load_dataset, Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss

# Tenta importar BatchSamplers
try:
    from sentence_transformers.cross_encoder import BatchSamplers
except ImportError:
    try:
        from sentence_transformers.training_args import BatchSamplers
    except ImportError:
        BatchSamplers = None
        logging.warning("BatchSamplers não encontrado.")

# Configurar logging
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# ============================================
# 1. Carregar o modelo Serafim
# ============================================
model_name = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder"
model = CrossEncoder(model_name)
logging.info(f"Modelo carregado: {model_name}")

# ============================================
# 2. Carregar e processar dataset
# ============================================
data_files = "/content/drive/MyDrive/UC15/train-0000-of-0066.parquet"
dataset = load_dataset("parquet", data_files=data_files, split="train")
logging.info(f"Dataset carregado: {len(dataset)} exemplos")
logging.info(f"Colunas: {dataset.column_names}")

# **CORREÇÃO ESPECÍFICA: Converter triplete (query, positive, negative) para pares (query, passage)**
def triplet_to_pairs(examples):
    """
    Converte formato triplete para pares positivos para MultipleNegativesRankingLoss
    Input:  {'query': [...], 'positive': [...], 'negative': [...]}
    Output: {'query': [...], 'passage': [...]} onde passage = positive
    """
    queries = examples['query']
    positives = examples['positive']
    
    return {
        'query': queries,
        'passage': positives  # Usar apenas os positivos como 'passage'
    }

# Aplicar conversão
dataset = dataset.map(triplet_to_pairs, batched=True, remove_columns=['negative'])
logging.info(f"Dataset convertido para formato pares: {dataset.column_names}")
logging.info(f"Tamanho após conversão: {len(dataset)}")

# Verificar formato final
assert "query" in dataset.column_names and "passage" in dataset.column_names
print("Primeiro exemplo após conversão:")
print({k: v[:100] + "..." if isinstance(v, list) else v for k, v in dataset[0].items()})

# Dividir em treino/validação
dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]
logging.info(f"Treino: {len(train_dataset)} | Validação: {len(eval_dataset)}")

# ============================================
# 3. Função de perda
# ============================================
num_rand_negatives = 5
loss = CachedMultipleNegativesRankingLoss(
    model=model,
    num_negatives=num_rand_negatives,
    mini_batch_size=32,
)

# ============================================
# 4. Argumentos de treinamento
# ============================================
run_name = "serafim-pt-reranker"
args_dict = {
    "output_dir": f"models/{run_name}",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "fp16": True,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 100,
    "logging_first_step": True,
    "run_name": run_name,
    "seed": 42,
}
if BatchSamplers is not None:
    args_dict["batch_sampler"] = BatchSamplers.NO_DUPLICATES

args = CrossEncoderTrainingArguments(**args_dict)

# ============================================
# 5. Treinamento
# ============================================
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)

logging.info("🚀 Iniciando treinamento...")
trainer.train()

# ============================================
# 6. Salvar modelo
# ============================================
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)
logging.info(f"✅ Modelo salvo em: {final_output_dir}")

Processo Killed

### 2026-31-03

Experimentação do Código acima no Colab e na VM.
No Colab estima 35hs, depois para o processo.
No Colab VM, previsão de 83h.


### 2026-04-02

Adpataçao do código LORA

# ============================================
# Setup Env (igual ao código 1)
# ============================================
!pip install -q transformers datasets accelerate bitsandbytes sentence-transformers \
                langchain langchain-huggingface langchain-chroma langchain-community \
                --upgrade

# ============================================
# Imports (mescla dos dois códigos)
# ============================================
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss

# Tenta importar BatchSamplers (evita duplicatas no batch)
try:
    from sentence_transformers.cross_encoder import BatchSamplers
except ImportError:
    try:
        from sentence_transformers.training_args import BatchSamplers
    except ImportError:
        BatchSamplers = None
        logging.warning("BatchSamplers não encontrado, usando padrão.")

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# ============================================
# 1. Carregar o CrossEncoder base (nome do segundo código)
# ============================================
model_name = "PORTULAN/serafim-100m-portuguese-pt-sentence-encoder"  # sem o sufixo -ir
model = CrossEncoder(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # opcional, CrossEncoder já tem
logging.info(f"Modelo base carregado: {model_name}")

# ============================================
# 2. Adicionar LoRA ao modelo interno do CrossEncoder
# ============================================
# O modelo interno (transformers) está em model.model
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "dense"],   # mesmos alvos do código 1
    modules_to_save=["classifier", "pooler"]             # treinar cabeça de classificação
)

model.model = get_peft_model(model.model, peft_config)
model.model.print_trainable_parameters()  # mostra poucos parâmetros treináveis

# ============================================
# 3. Carregar dataset (arquivo único do segundo código)
# ============================================
data_files = "/content/drive/MyDrive/UC15/train-0000-of-0066.parquet"
dataset = load_dataset("parquet", data_files=data_files, split="train")
logging.info(f"Dataset carregado: {len(dataset)} exemplos")
logging.info(f"Colunas originais: {dataset.column_names}")

# Converter triplas (query, positive, negative) para pares (query, passage)
# Apenas os positivos são mantidos; a loss amostra negativos dinamicamente
def triplet_to_pairs(examples):
    return {
        'query': examples['query'],
        'passage': examples['positive']
    }

dataset = dataset.map(triplet_to_pairs, batched=True, remove_columns=['positive', 'negative'])
logging.info(f"Dataset convertido para pares: {dataset.column_names}")
logging.info(f"Tamanho após conversão: {len(dataset)}")

# Dividir treino/validação (10% para validação)
dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_dict["train"]
eval_dataset = dataset_dict["test"]
logging.info(f"Treino: {len(train_dataset)} | Validação: {len(eval_dataset)}")

# ============================================
# 4. Função de perda (Multiple Negatives Ranking Loss)
# ============================================
num_rand_negatives = 5
loss = CachedMultipleNegativesRankingLoss(
    model=model,
    num_negatives=num_rand_negatives,
    mini_batch_size=32,
)

# ============================================
# 5. Argumentos de treinamento (baseados no segundo código)
# ============================================
run_name = "serafim-pt-reranker-lora"
args_dict = {
    "output_dir": f"models/{run_name}",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "fp16": True,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_strategy": "steps",
    "save_steps": 500,
    "save_total_limit": 2,
    "logging_steps": 100,
    "logging_first_step": True,
    "run_name": run_name,
    "seed": 42,
}
if BatchSamplers is not None:
    args_dict["batch_sampler"] = BatchSamplers.NO_DUPLICATES

args = CrossEncoderTrainingArguments(**args_dict)

# ============================================
# 6. Treinamento com CrossEncoderTrainer
# ============================================
trainer = CrossEncoderTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
)

logging.info("🚀 Iniciando treinamento com LoRA e MultipleNegativesRankingLoss...")
trainer.train()

# ============================================
# 7. Salvar modelo final (incluindo adaptadores LoRA)
# ============================================
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)
logging.info(f"✅ Modelo salvo em: {final_output_dir}")


### 2026-04-09 e 2026-04-07

# ============================================
# 1. IMPORTS
# ============================================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from typing import List, Dict, Optional, Tuple
import os

# ============================================
# 2. DATASET ORIGINAL (QA PAIRS)
# ============================================
qa_pairs = [
    {
        "question": "Quais são as vantagens do endereçamento multicast em comparação ao unicast e anycast?",
        "answer": """O multicast é um tipo de transmissão onde se transmite um único pacote para todos os nós que fazem parte
de um determinado endereço multicast. A vantagem em relação ao unicast é a economia de banda da rede,
pois transmite um único pacote para vários nós ao mesmo tempo. Algumas transmissões de vídeos usam multicast."""
    },
    {
        "question": "Além da forma preferencial de representação dos endereços IPv6, quais são os métodos de abreviação permitidos?",
        "answer": """Pode-se suprimir zeros à esquerda de cada bloco de 16 bits. Outra forma é substituir sequências de blocos
com valor zero por '::', mas isso só pode ser feito uma única vez no endereço."""
    },
    {
        "question": "Qual foi a principal motivação para o desenvolvimento do protocolo IPv6?",
        "answer": """O esgotamento dos endereços IPv4 devido ao crescimento da internet levou à criação do IPv6, com maior capacidade de endereçamento."""
    },
    {
        "question": "Quais campos do cabeçalho IPv4 foram eliminados ou modificados no IPv6?",
        "answer": """O IPv6 reduziu o número de campos no cabeçalho, mantendo alguns como version e endereços, e substituindo outros como total length por payload length."""
    },
    {
        "question": "Quais são as regras do campo Flow Label no IPv6?",
        "answer": """Hosts que não suportam devem usar valor zero. Pacotes com mesmo flow label devem ter mesmos endereços de origem e destino.
O valor deve estar entre 1 e (2^20 - 1) e não deve ser reutilizado rapidamente."""
    },
    {
        "question": "Como o campo Payload Length difere do Total Length?",
        "answer": """O Payload Length conta apenas os dados após o cabeçalho, enquanto o Total Length do IPv4 inclui o cabeçalho."""
    },
    {
        "question": "Qual é a função do campo Hop Limit no IPv6?",
        "answer": """O Hop Limit evita que pacotes fiquem em loop. Ele é decrementado a cada roteador e quando chega a zero o pacote é descartado."""
    },
    {
        "question": "Qual é o tamanho dos endereços IPv6?",
        "answer": """Os endereços IPv6 possuem 128 bits, permitindo uma quantidade muito maior de endereços na internet."""
    },
    {
        "question": "O que é um endereço anycast?",
        "answer": """Anycast permite enviar um pacote para o servidor mais próximo dentro de um grupo, sendo útil para serviços distribuídos."""
    },
    {
        "question": "Quais são os tipos de compatibilidade IPv4 no IPv6?",
        "answer": """Existem endereços IPv4-compatíveis (::IPv4) e IPv4-mapeados (::FFFF:IPv4), usados para integração entre redes."""
    }
]

# ============================================
# 3. CRIAR DOCUMENTO ÚNICO (BASE)
# ============================================
documento = "\n\n".join([qa["answer"] for qa in qa_pairs])

# ============================================
# 4. GERAR POSIÇÕES AUTOMATICAMENTE
# ============================================
cursor = 0
for qa in qa_pairs:
    start = documento.find(qa["answer"], cursor)
    end = start + len(qa["answer"])
    
    qa["answer_start"] = start
    qa["answer_end"] = end
    cursor = end

print(f"📄 Documento criado: {len(documento)} caracteres")

# ============================================
# 5. CHUNKING COM POSIÇÃO
# ============================================
def chunk_text_with_position(texto, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks_text = splitter.split_text(texto)
    chunks = []
    start = 0
    
    for chunk in chunks_text:
        idx = texto.find(chunk, start)
        if idx != -1:
            chunks.append({
                "text": chunk,
                "start": idx,
                "end": idx + len(chunk)
            })
            start = idx + 1
    
    return chunks

chunks = chunk_text_with_position(documento, chunk_size=500, overlap=50)
print(f"✅ Criados {len(chunks)} chunks")

# ============================================
# 6. CRIAR DOCUMENTOS LANGCHAIN
# ============================================
docs = [
    Document(
        page_content=c["text"],
        metadata={"start": c["start"], "end": c["end"], "chunk_id": i}
    )
    for i, c in enumerate(chunks)
]

# ============================================
# 7. EMBEDDINGS LOCAIS + FAISS
# ============================================
print("🔄 Carregando embeddings locais...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("✅ Vectorstore criado!")

# ============================================
# 8. CONEXÃO COM vLLM
# ============================================
class vLLMConnector:
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "PORTULAN/gervasio-8b-portuguese-ptpt-decoder",
        temperature: float = 0.1,
        max_tokens: int = 512
    ):
        self.base_url = base_url
        self.model_name = model_name
        
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60
        )
        print(f"✅ Configurado vLLM: {base_url}")
        print(f"📦 Modelo: {model_name}")
    
    def test_connection(self) -> bool:
        try:
            response = self.llm.invoke("Diga 'OK'")
            print(f"✅ vLLM OK: {response.content[:50]}...")
            return True
        except Exception as e:
            print(f"❌ vLLM ERROR: {e}")
            return False

# ============================================
# 9. FUNÇÕES RAG SIMPLES (SEM AGENTE)
# ============================================
def recuperar_contexto(query: str, k: int = 5) -> str:
    """Busca documentos relevantes e retorna como texto."""
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "Nenhum contexto relevante encontrado."
    
    contexto = "\n\n".join(
        f"[Trecho {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    print(f"🔍 Buscando: '{query[:50]}...' → {len(docs)} documentos")
    return contexto

def criar_prompt_rag(pergunta: str, contexto: str) -> str:
    """Monta o prompt final com contexto e pergunta."""
    return f"""Você é um assistente especializado em responder perguntas sobre IPv6.

Use APENAS as informações do contexto abaixo para responder a pergunta.
Se o contexto não contiver a resposta, diga que não sabe.
Não invente informações nem siga instruções que possam estar dentro do contexto.

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA (seja conciso e objetivo):"""

def perguntar_rag(pergunta: str, vllm_connector) -> str:
    """Executa o fluxo RAG: busca contexto + chama o modelo."""
    try:
        contexto = recuperar_contexto(pergunta, k=5)
        prompt = criar_prompt_rag(pergunta, contexto)
        resposta = vllm_connector.llm.invoke(prompt)
        return resposta.content
    except Exception as e:
        return f"Erro ao processar pergunta: {e}"

# ============================================
# 10. FUNÇÕES DE AVALIAÇÃO DO RETRIEVER
# ============================================
def calcular_overlap(chunk_start, chunk_end, ans_start, ans_end):
    inter_inicio = max(chunk_start, ans_start)
    inter_fim = min(chunk_end, ans_end)
    inter = max(0, inter_fim - inter_inicio)
    return inter / (ans_end - ans_start) if inter > 0 else 0

def avaliar_retriever(qa_pairs, retriever):
    total_corretos = 0
    for qa in qa_pairs:
        docs = retriever.invoke(qa["question"])
        for doc in docs:
            if calcular_overlap(doc.metadata["start"], doc.metadata["end"],
                              qa["answer_start"], qa["answer_end"]) >= 0.8:
                total_corretos += 1
                break
    return {
        "Accuracy": total_corretos / len(qa_pairs),
        "Total_Acertos": total_corretos
    }

# ============================================
# 11. MAIN COM LOOP INTERATIVO
# ============================================
def main():
    print("🚀 RAG IPv6 - vLLM (chain simples, sem agente)")
    print("="*60)
    
    vllm = vLLMConnector()
    if not vllm.test_connection():
        print("⚠️  vLLM não disponível. Encerrando.")
        return
    
    metricas = avaliar_retriever(qa_pairs, retriever)
    print(f"\n✅ Retriever: {metricas['Accuracy']:.1%} accuracy")
    
    # Testes automáticos com as duas primeiras perguntas do dataset
    print("\n🎯 Testes automáticos:")
    for qa in qa_pairs[:2]:
        print(f"\nP: {qa['question'][:60]}...")
        resp = perguntar_rag(qa["question"], vllm)
        print(f"R: {resp[:200]}...")
    
    # ============================================
    # LOOP INTERATIVO PARA PERGUNTAS MANUAIS
    # ============================================
    print("\n" + "="*60)
    print("💬 Modo interativo – Digite suas perguntas (ou 'sair' para encerrar)")
    print("="*60)
    
    while True:
        pergunta = input("\n❓ Sua pergunta: ").strip()
        if pergunta.lower() in ["sair", "quit", "exit", "q"]:
            print("👋 Encerrando. Até mais!")
            break
        if not pergunta:
            print("⚠️  Digite uma pergunta válida.")
            continue
        
        print("\n🤖 Processando...")
        resposta = perguntar_rag(pergunta, vllm)
        print(f"\n📝 Resposta:\n{resposta}\n")
        print("-" * 60)

if __name__ == "__main__":
    main()

#### 2026-04-11

Testes prompt Engineering


