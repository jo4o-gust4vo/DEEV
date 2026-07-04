import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregar os dados (com as colunas corretas que descobrimos antes)
df_deev = pd.read_csv(r'D:\Documentos\Estudos\Mestrado\Aprendizado Federado\_DEEV\DEEV\logs\MNIST\None-POC-0.2\DNN\server.csv', header=None, names=['timestamp', 'round', 'loss', 'accuracy', 'top5'])

# 2. ORDENAR AS RODADAS
# Isso impede que a linha volte para trás no gráfico
df_deev = df_deev.sort_values(by='round')

# 3. APLICAR A MÉDIA MÓVEL (A Suavização)
# O "window=5" tira a média a cada 5 rodadas. Você pode aumentar para 10 se quiser a linha ainda mais reta.
df_deev['accuracy_suave'] = df_deev['accuracy'].rolling(window=5, min_periods=1).mean()

# 4. Configurar a figura
plt.figure(figsize=(10, 6))

# (Opcional) Plotar os dados reais "crus" no fundo, bem transparentes (alpha=0.2)
plt.plot(df_deev['round'], df_deev['accuracy'], color='#2ca02c', alpha=0.2, linestyle='-')

# Plotar a linha suavizada principal (grossa e sem as bolinhas, igual ao artigo)
plt.plot(df_deev['round'], df_deev['accuracy_suave'], label='DEEV', color='#2ca02c', linewidth=2, linestyle='-')

# 5. Estilizar
plt.xlabel('Rodada de Comunicação (#)', fontsize=12)
plt.ylabel('Acurácia (%)', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(loc='lower right', fontsize=12)

# Ajuste do limite Y (opcional, mude conforme seus dados)
plt.ylim(0.3, 1.05)

# Salvar e mostrar
plt.tight_layout()
plt.savefig('grafico_deev_final.png', dpi=300)
plt.show()