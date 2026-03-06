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


