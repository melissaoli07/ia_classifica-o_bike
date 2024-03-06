from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Aqui estamos carregando o modelo já treinado que está no arquivo JobLib
with open('C:/Users/HP/Desktop/fiap/aichatbot/sprint4/meu_modelo_serializado.pickle', 'rb') as f:
    modelo = pickle.load(f)

# Rota para receber os dados e fazer previsões
@app.route('/prever', methods=['GET'])
def prever():
    # Obter parâmetros da solicitação GET
    parametro1 = float(request.args.get('marca_da_bike'))
    parametro2 = float(request.args.get('modelo_da_bike'))
    parametro3 = float(request.args.get('valor'))

    # Fazer previsões usando o modelo 
    entrada = np.array([[parametro1, parametro2, parametro3]])
    resultado = modelo.predict(entrada)

    # Retornar o resultado como JSON
    return jsonify({'previsao do tipo da bicicleta': resultado.tolist()})

if __name__ == '__main__':
    print("Servidor Flask em execução")
    # Executar o aplicativo Flask
    app.run(debug=True)
