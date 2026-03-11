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