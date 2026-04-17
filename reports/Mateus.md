## Diário

### 2026-03-05

Foi definido que o material de base para treinamento do modelo será o arquivo "TCC versão 4.6.pdf", o qual já está em PDF.

### 2026-03-06

PyMuPDF foi testado e obtivemos sucesso ao transformar PDF em texto.


### 2026-03-19

Voltei hoje, ainda tenho que me atualizar em algumas coisas do projeto. Testei a VM e não obtive sucesso

### 2026-03-24
VM finalmente funcionando, estava no processo de instalar firefox.

### 2026-03-26
O que é o Freenet6?
R = O Freenet6 é um serviço de acesso IPv6 do tipo Tunnel Broker oferecido gratuitamente a comunidade que
permite a milhares de pessoas o acesso simultâneo a conectividade IPv6 utilizando como infraestrutura a
Internet IPv4.

Como é a infraestrutura do Freenet6?
R = A infraestrutura do Freenet6 é baseada no produto Gateway6 da Hexago, o qual provê endereços e prefixos
IPv6 fixos, registro DNS e delegação reversa. Os usuários do Freenet6 podem se conectar a partir de
qualquer conexão quê permita acesso a Internet, mesmo estando sob um NAT.

Como é o Registro de DNS no modo autenticado?
R = No modo Autenticado, o FQDN associado aos túneis autenticados usam o formato
userid.broker.freenet6.net, onde userid é o nome da conta do usuário. Um usuário registrado como
“exemploipv6” teria a seguinte entrada DNS: exemploipv6.broker.freenet6.net.

Como é o Registro de DNS no modo anônimo?
R = No modo anônimo, o FQDN associado aos túneis anônimos usa o formato
“endereço_IPv4_do_anônimo.servidor.freenet6.net”, onde endereço_IPv4_anônimo é o endereço IPv4 do
cliente anônimo. Por exemplo, um usuário conectado a partir do endereço IPv4 206.123.31.231 teria a
seguinte entrada DNS: anonymous-206.123.31.231.tsps2.freenet6.net (tsps2 é um dos servidores do
Freenet6).

Como o Freenet6 pode ser utilizado?
R = O Freenet6 pode ser utilizado de duas formas: autenticado ou anônimo. No modo autenticado, o endereço e
prefixo IPv6 são designados permanente para um usuário e não mudam mesmo que o endereço IPv4 do
usuário mude. No modo anônimo, o endereço IPv6 muda caso haja mudança do endereço IPv4.

Qual tempo de vida dos túneis?
R = A menos que sejam renovados, os túneis padrões expiram em sete dias depois da última conexão do cliente
Freenet6. 

Qual a diferença entre o Freenet6 e outros tunnel brokers?
R = Diferentemente de outros tunnel brokers convencionais que disponibilizam uma interface web, o Freenet6
utiliza um modelo baseado na arquitetura cliente/servidor. O cliente Gateway6 é um software que é
executado em um computador implementando o Tunnel Setup Protocol (TSP). Esse cliente negocia
automaticamente um túnel configurado entre o computador ou roteador e o Tunnel Broker Freenet6,
tornando a implementação de conectividade IPv6 simples de se instalar e manter, além de ser gratuito,
licenciado através da GPL

### 2026-03-27
GPU funcionando na maquina virtual, docker esta instalado e funcionando com a GPU. Falta apenas rodar os arquivos docker para instalar Marimo e Doccano

### 2026-03-31
Rodei o modelo usando o codigo que o professor passou na VM

### 2026-04-07
Instalaçao vLLM, falta integrar com fast api 

### 2026-04-09
Integração com fast api sendo feita, Weverton já fez e esta testando o modelo. ate agora o resultado é satisfatório.

### 2026-04-13
falta rodar o codigo e pesquisei sobre o que o professor passou na aula.

### 2026-04-14
revisão do codigo

### 2026-04-16
fazendo testes
