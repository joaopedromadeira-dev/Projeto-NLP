# 05/03/26 - João Pedro

## Base de dados foi decidida em decisão unânime pelo grupo e foi upada ao github em formato PDF

# 06/03/26 - João Pedro

## Código PyMuPDF library utilizado para extrair textos de PDF, https://pymupdf.readthedocs.io/en/latest/the-basics.html

# 09/03/26

## Confecção de 10 perguntas baseadas no PDF que escolhemos como base de dados para treino do modelo

### 10 perguntas 
1.Qual a técnica que permite a coexistência de IPv4 e Ipv6 nos mesmos dispositivos e redes?
R=Técnica de Pilha dupla (Dual-stack), que permitem a coexistência de IPv4 e IPv6 nos mesmos
dispositivos e redes.

2.Como é denominada os conjuntos de técnicas que permitem o transporte de tráfego IPv6 na infra-estrutura IPv4 existente?
R=Técnicas de Tunelamento, que permitem o transporte de tráfego IPv6 na infra-estrutura IPv4
existente.

3.Como se chama as técnicas que permitem a "nós apenas IPv6" comunicarem com "nós apenas Ipv4" ?
R=Técnicas de tradução, que permitem a “nós apenas IPv6” comunicarem com nós Apenas-IPv4

4.Como se comporta um "nó IPv6/IPv4"?
R=Um nó que utilize pilha dupla tem suporte complete para as duas versões do protocolo IP. Este tipo de nó é
geralmente citado como um “nó IPv6/IPv4”. Em comunicação com um nó IPv6, o “nó IPv6/IPv4” se comporta
como um “nó apenas-IPv6”, e em comunicação com um nó IPv4, ele se comporta como um “nó apenasIPv4”. Prováveis implementações de nós como estes (“nós IPv6/IPv4”) possivelmente alternarão seu
comportamento, hora acionando uma pilha, hora outra. Estes tipos de nós podem apresentar até três modos
de operação. Quando a pilha IPv4 está acionada e a pilha IPv6 não está acionada, o nó se comporta como
um “nó apenas IPv4”. Quando a pilha IPv6 está acionada e a pilha IPv4 não está acionada, o nó se
36
comporta como um “nó apenas IPv6”. Quando as duas pilhas estão funcionando, o nó pode utilizar ambos
os protocolos. Um “nó IPv6/IPv4” tem pelo menos um endereço para cada versão do protocolo. Ele utiliza
mecanismos IPv4 para ser conFigurado para um endereço IPv4 (conFiguração estática ou DHCP) e usa
mecanismos IPv6 para ser conFigurado para um endereço IPv6 (conFiguração estática ou automática -
“autoconFiguration”).

5.Como os mecanismos de tunelamento são utilizados?
R=Os mecanismos de tunelamento podem ser utilizados como ponto de partida para uma infra-estrutura de
encaminhamento (“forwarding”) IPv6 enquanto a infra-estrutura IPv4 permanece como a base. O
tunelamento pode ser utilizado para carregar tráfego IPv6 através do encapsulamento em pacotes IPv4 para
que estes trafeguem na infra-estrutura IPv4 de roteamento. Por exemplo, se seu provedor de Internet possui
uma infra-estrutura “apenas IPv4”, o tunelamento permite que você tenha uma rede corporativa IPv6,
atravesse seus dados (IPv6) através da rede “apenas IPv4” do seu provedor de Internet e alcance outras
redes ou hosts IPv6.

6.Quantos e quais tipos de tunelamento existem?
R=– Tunelamento de IPv6 sobre IPv4 conFigurado manualmente:
Os pacotes IPv6 são encapsulados em pacotes IPv4 para serem carregados através das
infra-estruturas de roteamento IPv4. Esses túneis são ponto-a-ponto que necessitam de
conFiguração manual.
– Tunelamento automático de IPv6 sobre IPv4:
Nós IPv6 podem usar diferentes tipos de endereços, como endereços IPv6 compatíveis com
IPv4, endereços “6to4” ou endereços ISATAP, para promover o tunelamento dinâmico dos
pacotes IPv6 através de uma infra-estrutura de roteamento IPv4. Esses endereços IPv6
unicast especiais carregam um endereço IPv4 em algum dos campos de endereço IPv6.

7.Como funciona o tunelamento geral?
R=O host "A" está numa rede IPv6 e quer enviar um pacote IPv6 para o host "B" em outra rede IPv6. A rede
entre os roteadores R1 e R2 é uma rede “apenas IPv4”. O roteador R1 é o ponto de entrada do túnel . O
host "A" envia o pacote IPv6 para o roteador R1 (Passo 1 na Figura 4.1). Quando o roteador R1 recebe o
pacote endereçado a "B", ele encapsula o pacote com um cabeçalho IPv4 e o encaminha para o roteador
R2 (Passo 2 na Figura 4.1), o qual é o ponto de saída do túnel . O roteador R2 desencapsula o pacote e o
encaminha para seu destino final (Passo 3 na Figura 4.1). Entre R1 e R2 pode haver um infinito número de
roteadores IPv4.

8.O que é NAT e como funciona?
R=Tradução de Endereço de Rede (NAT)
O NAT tem sido muito utilizado, especialmente para sobrepor as limitações espaciais do endereçamento
IPv4. Redes corporativas usam endereços IPv4 da faixa privada e um roteador NAT na borda da rede
corporativa para traduzir os endereços privados em um único ou em alguns poucos endereços públicos.
49
O NAT, como descrito aqui, prove roteamento entre uma rede IPv6 e uma rede IPv4. Na sua forma mais
básica do NAT, um bloco de endereços IPv4 é reservado e os campos de endereço IP de origem, IP e os
cabeçalhos de checagem de IP, TCP, UDP, e ICMP são traduzidos. Com o NAPT, outros identificadores de
transporte como números de porta TCP e UDP e tipos de mensagens ICMP são traduzidos também. Isso
permite a um conjunto de hosts IPv6 compartilhar um único endereço IPv4.
O NAT pode ser unidirecional (apenas hosts IPv6 podem iniciar uma sessão) ou bidirecional (a sessão pode
ser iniciada de ambos os lados). Hosts na rede IPv4 utilizam o DNS para resolução de nomes. Além disso,
um Gateway de Camada de Aplicação DNS (ALG) pode prover a tradução de endereços IPv6 em seus
endereços combinados IPv4 NAT e vice-versa.

9.Como funciona a tradução de pacotes?
R=Para entender como os pacotes são traduzidos, nós seguiremos um pacote sendo enviado de um host IPv6
através de um gateway NAPT para um host IPv4 e de volta. A Figura 4.10 ilustra o processo de tradução.Neste exemplo, A, o host IPv6, tem um endereço IPv6 ABCD:BEEF::2228:7001. B, do outro lado do
roteador NAPT, tem um endereço IPv4 120.140.160.101. O gateway NAPT foi configurado com a faixa
50
120.10.40/24. A inicializa a sessão com B enviando um pacote para o endereço de destino
prefix::120.140.160.101, porta 23. O prefixo ::/96 é divulgado pelo NAPT na rede IPv6, e sempre que um
pacote for enviado para este prefixo, este será roteado através do NAPT. Como endereço de origem, A usa
seu endereço IPv6 com o número de porta 3056 (passo 1).
O gateway NAPT designa um endereço IPv4 e um número de porta dentro da faixa que lhe foi configurada.
Vamos considerar que ele utilize o endereço 120.10.40.10. O novo pacote IPv4 que será enviado do NAPT
para B tem um endereço de origem igual a 120.10.40.10, porta 1025, e um endereço de destino igual a
120.140.160.101, porta 23 (passo 2).
Quando B responde, ele envia o pacote com um endereço de origem igual a 120.140.160.101, porta 23,
para o endereço de destino 120.10.40.10, porta 1025 (passo 3). O NAPT traduz o pacote de acordo com os
parâmetros que tem armazenado em seu cache durante toda a sessão e envia os pacotes e os envia com o
endereço de origem prefix::120.140.160.101, porta 23, para o endereço de destino ABCD:BEEF::2228:7001,
porta 3056 (passo 4)

10.Como os mecanismos de tradução são classificados?
R=Os mecanismos de tradução, explicados anteriormente, podem ser distribuídos conforme segue abaixo:
– Tradução (rede) - é o método de tradução, que ocorre na camada de rede. Os mecanismos, que
fazem parte desta classe são SIIT, NAT-PT e NAPT-PT;
– Tradução (transporte) - é o método de tradução, que ocorre na camada de transporte. Os
mecanismos, que integram esta classe, são TRT e SOCKS;
– Tradução (aplicação) - é o método de tradução, que ocorre na camada de aplicação. O mecanismo,
que constitue esta classe, é o ALG;
– Tradução (camada adicional) - é o método de tradução, que recebe a adicão de uma nova camada
na pilha de protocolos. Os mecanismos, que fazem parte desta classe, são BIS e BIA.

### Perguntas Extras
1.É possível fazer a transição para IPv6 enquanto o provedor continua usando IPv4?R = “Você pode fazer a transição em sua rede inteira, ou partes dela, enquanto seu servidor de Internet continua funcionando apenas com IPv4.” 
2.O que descreve a RFC 2893?R = “A RFC 2893, ‘Transition Mechanisms for IPv6 Hosts and Routers’, descreve o conjunto inicial de mecanismos de transição.”
3.O que é um nó que utiliza pilha dupla?R = “Um nó que utilize pilha dupla tem suporte complete para as duas versões do protocolo IP.”
4.Como redes corporativas geralmente utilizam NAT?R = “Redes corporativas usam endereços IPv4 da faixa privada e um roteador NAT na borda da rede corporativa para traduzir os endereços privados em um único ou em alguns poucos endereços públicos.” 
