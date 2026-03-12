## Diário

### 2025-03-05

* Foi definido que o material de base para treinamento do modelo será o arquivo "TCC versão 4.6.pdf"

### 2025-03-06

* Testes nas bibliotecas PyMuPDF e PyPDF. A primeira deu um resultado muito mais satisfatório.

Código de exemplo:

```py
doc = pymupdf.open("TCC  versao 4.6.pdf") # open a document
_out = open("TCC  versao 4.6.txt", "wb") # create a text output
for _page in doc: # iterate the document pages
    _text = _page.get_text().encode("utf8") # get plain text (is in UTF-8)
    _out.write(text) # write text of page
    _out.write(bytes((12,))) # write page delimiter (form feed 0x0C)
_out.close()
```

* Pacotes demonstrados

```py
from sentence_transformers impor SentenceTransformer
import chromadb
import hashlib
```

* Modelos: https://huggingface.co/PORTULAN

### 2025-03-09

#### Information Retrieval

* MLOps
* Embedding
  * Sentence-Transformers
* VectorDB
  * ChromaDB
  * QDrant
* Orchestrator
  * Langchain 
* Backend API
  * FastAPI
* UI Framework
  * StreamLit

#### Tarefa do dia

* Elaborar um questionário sobre o material a ser usado no treinamento. 10 perguntas por aluno
* Antônio: Capítulo 2
* João Pedro: Capítulo 4
* Weverton: Capítulos 1 e 3

##### Questões

||
| --- |
| 1. No que tange ao endereçamento, qual a principal diferença entre o IPv4 e o IPv6? |
| R: O protocolo IPv4 utiliza apenas 32 bits para o  endereçamento e uma divisão em classes ineficiente, o que vem tornando o roteamento cada vez mais complexo devido ao crescente número de hosts e os endereço IPv4 cada vez mais escassos. Para solucionar tal problema, o protocolo IPv6 utiliza 128 bits para o endereçamento, aumentando o número de endereços possíveis de 232 para 6,65 x 1023, 1500 endereços ipv6 por metro quadrado da superfície do planeta Terra, solucionando o problema de espaço de endereçamento e simplificando as tabelas de roteamento. |
| 2. As duas versões de protocolo poderão coexistir? |
| R: É fato que as duas versões do protocolo IP devam conviver por tempo indefinido, e para garantir uma boa convivência, o protocolo IPv6 foi desenvolvido com mecanismos que permitam a interoperabilidade em um ambiente misto. |
| 3. Quais os principais cabeçalhos de extensão do IPv6? |
| R: Identificador - Extended Options Header <br>0 - Hop-by-hop Options Header <br> 43 - Routing Information <br> 44 - Fragment <br> 51 - Authentication Header <br> 50 - Encapsulating Security Payload Header <br> 59 - No Next Header <br> 60 - Destination Options Header | 
| 4. Ao usar a opção de Jumbo Payload, qual o tamanho máximo suportado em MB ? |
| R: é possível representar um pacote com tamanho entre 65.535 a 4.294.967.295 bytes. |
| 5. Qual a vantagem de fragmentar o pacote somente na origem ? |
| R: Na nova versão, somente se fragmenta um pacote na origem, ou seja, não é mais possível fragmentar um pacote IPv6 nos nós intermediários, diminuindo o processamento do pacote pelo roteador.|
| 6. Qual campo define os endereços das rotas que o pacote deve percorrer ?|
| R: O campo address possui tamanho variável múltiplo de 128 bits, este campo irá definir os endereços das rotas no qual o pacote deverá percorrer|
| 7. Em que condição é necessário fragmentar um pacote na transmissão ?|
| R: Quando o tamanho do pacote for muito grande para ser transmitido pela rede, é necessário fragmentá-lo, 
podendo então ser transmitido. | 
| 8. O Authentication Header impede que a informação que está sendo transmitida seja descoberta em caso de análise da rede? | 
| R: O Authentication Header assegura quem está enviando o pacote e que o mesmo não teve sua rota alterada. Mas isto não quer dizer que alguém que esteja analisando a rede possa descobrir que informações estão passando. |
| 9. Quais campos formam o cabeçalho de opções Hop-by-hop|
| R: existem três campos que formam este cabeçalho, e eles serão explicados a seguir [DEE 98]:<br>O campo Next Header possui 8 bits de tamanho e identifica o próximo cabeçalho, usando os mesmos valores da tabela 3.1. Este campo aparece também em todos os outros cabeçalho de extensão. <br>O campo Header Extended Lenght representa o tamanho do cabeçalho em unidades de 64 bits, sendo que os primeiros 64 bits não são incluídos. O campo Header Extended Lenght possui oito bits de tamanho [DEE 98]. <br>O campo Options define o tipo de informação que o Hop-by-Hop Options Header irá carregar. Este campo possui tamanho variável e pode ter de uma ou mais definições, ou tipo de informação.|
| 10. Quantas vezes os cabeçalhos opcionais de extensão podem aparecer num pacote IPv6? |
| R: eles podem aparecer mais de uma vez no pacote. O único cabeçalho que pode aparecer no máximo duas vezes é o Destination Options Header |

### 2025-03-10

#### Tarefa do dia

Testar sentence transformers (https://sbert.net)

#### Models

##### PORTULAN/serafim-900m-portuguese-pt-sentence-encoder

```
Query: Quantas vezes os cabeçalhos opcionais de extensão podem aparecer num pacote IPv6
Top 5 most similar sentences in answears:
(Score: 0.7303) eles podem aparecer mais de uma vez no pacote. O único cabeçalho que pode...
(Score: 0.6436) É fato que as duas versões do protocolo IP devam conviver por tempo...
(Score: 0.6323) O protocolo IPv4 utiliza apenas 32 bits para o endereçamento e uma divisão em...

Query: No que tange ao endereçamento, qual a principal diferença entre o IPv4 e o IPv6
Top 5 most similar sentences in answears:
(Score: 0.6793) O protocolo IPv4 utiliza apenas 32 bits para o endereçamento e uma divisão em...
(Score: 0.6292) É fato que as duas versões do protocolo IP devam conviver por tempo...
(Score: 0.5394) O Authentication Header assegura quem está enviando o pacote e que o mesmo...

Query: As duas versões de protocolo poderão coexistir
Top 5 most similar sentences in answears:
(Score: 0.7297) É fato que as duas versões do protocolo IP devam conviver por tempo...
(Score: 0.6722) eles podem aparecer mais de uma vez no pacote. O único cabeçalho que pode...
(Score: 0.5560) O Authentication Header assegura quem está enviando o pacote e que o mesmo...
```

##### alfaneo/bertimbau-base-portuguese-sts

```
Query: Quantas vezes os cabeçalhos opcionais de extensão podem aparecer num pacote IPv6
Top 5 most similar sentences in answears:
(Score: 0.7143) eles podem aparecer mais de uma vez no pacote. O único cabeçalho que pode...
(Score: 0.5926) existem três campos que formam este cabeçalho, e eles serão explicados a...
(Score: 0.5348) O protocolo IPv4 utiliza apenas 32 bits para o endereçamento e uma divisão em...

Query: No que tange ao endereçamento, qual a principal diferença entre o IPv4 e o IPv6
Top 5 most similar sentences in answears:
(Score: 0.6407) O protocolo IPv4 utiliza apenas 32 bits para o endereçamento e uma divisão em...
(Score: 0.6125) É fato que as duas versões do protocolo IP devam conviver por tempo...
(Score: 0.4761) eles podem aparecer mais de uma vez no pacote. O único cabeçalho que pode...

Query: As duas versões de protocolo poderão coexistir
Top 5 most similar sentences in answears:
(Score: 0.7310) É fato que as duas versões do protocolo IP devam conviver por tempo...
(Score: 0.6135) O protocolo IPv4 utiliza apenas 32 bits para o endereçamento e uma divisão em...
(Score: 0.5147) eles podem aparecer mais de uma vez no pacote. O único cabeçalho que pode...

```

### 2025-03-11

Introdução ao langchain e testes de uso