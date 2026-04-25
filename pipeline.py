import pandas as pd
import joblib
import numpy as np
import os

def prever_precos(caminho_arquivo_teste):
    """
    Função obrigatória para o corretor automático.
    Lê o arquivo de teste, carrega o modelo e retorna as predições.
    
    Parâmetros:
    caminho_arquivo_teste (str): Caminho local para o arquivo CSV de teste.
    
    Retorna:
    np.array ou pd.Series: As predições de preços.
    """
    # 1. Leitura dos dados de teste
    df_teste = pd.read_csv(caminho_arquivo_teste)

    # 2. Carregamento do modelo
    # Nota para o aluno: Se você mudar o nome do arquivo do modelo, atualize a linha abaixo.
    caminho_modelo = 'modelo_baseline.joblib'
    
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"O arquivo do modelo '{caminho_modelo}' não foi encontrado na raiz do projeto.")
        
    modelo = joblib.load(caminho_modelo)

    # 3. Predição
    # Certifique-se que o seu modelo/pipeline já trata as colunas necessárias.
    predicoes = modelo.predict(df_teste)

    return predicoes

if __name__ == "__main__":
    # Teste local: permite que o aluno valide o código antes da entrega.
    arquivo_teste_exemplo = 'teste_publico.csv'
    
    print(f"--- Executando Validação Local do Pipeline ---")
    
    if not os.path.exists(arquivo_teste_exemplo):
        print(f"[Aviso] Arquivo '{arquivo_teste_exemplo}' não encontrado.")
        print(f"Crie um arquivo CSV fictício com este nome para testar o script localmente.")
    else:
        try:
            resultados = prever_precos(arquivo_teste_exemplo)
            print("\n✅ Sucesso! O pipeline rodou corretamente.")
            print("Primeiras 5 predições:")
            print(resultados[:5])
        except Exception as e:
            print(f"\n❌ Erro encontrado no pipeline:")
            print(str(e))
            print("\nVerifique se o seu modelo espera as mesmas colunas presentes no CSV de teste.")
