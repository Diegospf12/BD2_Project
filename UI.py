from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QScrollBar, QFrame
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtCore import Qt, QPropertyAnimation
from PyQt5.QtGui import QPixmap

import psycopg2
import time
import sys
import pandas as pd
import csv
from spimi import TextRetrival, LoadData

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Configurar la ventana
        self.setWindowTitle("Super_proyecto_BD2")
        self.setGeometry(100, 100, 1500, 900)

        # Configurar el fondo
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 1500, 900)
        pixmap = QPixmap('fondo.jpg')
        self.label.setPixmap(pixmap.scaled(self.label.size(), QtCore.Qt.IgnoreAspectRatio))

       # Cargar la imagen del logo
        logo = QtGui.QPixmap('logo.png')
        logo = logo.scaled(300, 300, QtCore.Qt.KeepAspectRatio)

        # Configurar el layout
        self.layout = QGridLayout()
        self.widget = QtWidgets.QWidget(self)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

        # Configurar los elementos
        self.entrada = QLineEdit(self)
        self.entrada.setStyleSheet("QLineEdit {background-color: white; color: grey; font-size: 16px; font-weight: bold; text-shadow: 2px 2px 4px #000000; }")
        self.entrada.setPlaceholderText("Inserte el texto a buscar")
        self.boton_buscar = QPushButton("Buscar", self)
        self.boton_buscar.setStyleSheet("QPushButton { color: #000000; font-size: 16px; font-weight: bold; text-shadow: 2px 2px 4px #000000; }")
        self.boton_buscar.clicked.connect(self.buscar)
        self.entrada_numero = QLineEdit(self)
        self.entrada_numero.setStyleSheet("QLineEdit {background-color: white; color: grey; font-size: 16px; font-weight: bold; text-shadow: 2px 2px 4px #000000; }")
        self.entrada_numero.setPlaceholderText("Ingrese la cantidad de resultados esperados")

        self.cuadro1 = QLabel("Postgres: ", self)
        self.cuadro1.setStyleSheet("QLabel { color: #ffff00; font-size: 20px; font-weight: bold; text-shadow: 2px 2px 4px #00ffff; background-color: #000000; }")

        self.cuadro2 = QLabel("Resultado 2: ", self)
        self.cuadro2.setStyleSheet("QLabel { color: #ffff00; font-size: 20px; font-weight: bold; text-shadow: 2px 2px 4px #ffff00; background-color: #000000; }")

        self.resultados_text1 = QTextEdit(self)
        self.resultados_text1.setStyleSheet("QTextEdit { color: #ffff00; font-size: 16px; font-weight: bold; text-shadow: 2px 2px 4px #00ffff; background-color: #000000; }")
        self.resultados_text2 = QTextEdit(self)
        self.resultados_text2.setStyleSheet("QTextEdit { color: #ffff00; font-size: 16px; font-weight: bold; text-shadow: 2px 2px 4px #ffff00; background-color: #000000; }")

        self.scrollbar1 = QScrollBar(self)
        self.scrollbar1.setStyleSheet("QScrollBar { background-color: #000000; } QScrollBar::handle { background-color: #ff00ff; }")
        self.scrollbar2 = QScrollBar(self)
        self.scrollbar2.setStyleSheet("QScrollBar { background-color: #000000; } QScrollBar::handle { background-color: #00ff00; }")

        self.resultados_text1.setVerticalScrollBar(self.scrollbar1)
        self.resultados_text2.setVerticalScrollBar(self.scrollbar2)

        # Crear los QFrames para los cuadros
        frame_cuadro1 = QFrame(self)
        frame_cuadro1.setObjectName("frame_cuadro")
        frame_cuadro2 = QFrame(self)
        frame_cuadro2.setObjectName("frame_cuadro")

        # Configurar los estilos CSS para los QFrames
        frame_cuadro1.setStyleSheet("#frame_cuadro { background-color: #000000; border: 1px solid black; }")
        frame_cuadro2.setStyleSheet("#frame_cuadro { background-color: #000000; border: 1px solid black; }")

        # Configurar los layouts para los QFrames
        layout_cuadro1 = QVBoxLayout(frame_cuadro1)
        layout_cuadro2 = QVBoxLayout(frame_cuadro2)

        # Añadir los QLabel a los layouts de los QFrames
        layout_cuadro1.addWidget(self.cuadro1)
        layout_cuadro2.addWidget(self.cuadro2)

        # Añadir los QFrames al layout principal
        self.layout.addWidget(frame_cuadro1, 1, 0)
        self.layout.addWidget(frame_cuadro2, 1, 2)

        # Añadir los elementos al layout
        self.layout.addWidget(self.entrada, 0, 0)
        self.layout.addWidget(self.entrada_numero, 0, 1)  # Cambiado de posición
        self.layout.addWidget(self.boton_buscar, 0, 2)  # Cambiado de posición
        self.layout.addWidget(self.resultados_text1, 2, 0)
        self.layout.addWidget(self.resultados_text2, 2, 2)
        self.layout.addWidget(self.scrollbar1, 2, 1)
        self.layout.addWidget(self.scrollbar2, 2, 3)

        # Configurar la imagen del logo
        self.label_logo = QLabel(self)
        self.label_logo.setPixmap(logo)
        self.label_logo.setAlignment(QtCore.Qt.AlignCenter)

        # Añadir el QLabel al layout en la posición deseada
        self.layout.addWidget(self.label_logo, 1, 1)

        # Animación de desvanecimiento para el logo
        self.fade_animation = QPropertyAnimation(self.label_logo, b"opacity")
        self.fade_animation.setDuration(2000)
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.0)
        self.fade_animation.setEasingCurve(QtCore.QEasingCurve.InOutQuad)
        self.fade_animation.setLoopCount(-1)
        self.fade_animation.start()

    def buscar(self):
        texto_busqueda = self.entrada.text()

        conexion = psycopg2.connect(
            host="localhost",
            database="postgres",
            user="postgres",
            password="1234"
        )
        cursor = conexion.cursor()

        consulta = f"SELECT * FROM proyecto WHERE datos ILIKE '%{texto_busqueda}%'"

        start_time = time.time()
        cursor.execute(consulta)
        resultados = cursor.fetchall()
        execution_time = time.time() - start_time


        conteo_resultados = len(resultados)

        cuadro1_texto = f"Postgres: {conteo_resultados} resultados" +"\n" + f"tiempo de ejecución: {execution_time} msec"
        self.cuadro1.setText(cuadro1_texto)

        self.resultados_text1.clear()
        self.resultados_text2.clear()

        tiempo2 = 0
        resultados2 = []
        data = LoadData('styles.csv').get_data()

        if self.entrada_numero.text() != "":
            k = int(self.entrada_numero.text())
            tiempo2, resultados2 = TextRetrival().k_means(texto_busqueda, k, data)
        else:
            k = 10
            tiempo2, resultados2 = TextRetrival().k_means(texto_busqueda, k, data)

        for i, fila in enumerate(resultados):
            if i < k:
                self.resultados_text1.append(f"Resultado {i+1}: {fila}\n")

        for i, fila in enumerate(resultados2):
            if i < k:
                self.resultados_text2.append(f"Resultado {i+1}: {fila}\n")


        cuadro2_texto = f"tiempo de ejecución: {tiempo2} msec"
        self.cuadro2.setText(cuadro2_texto)

        cursor.close()
        conexion.close()

def index_csv_to_postgres(csv_file, table_name, db_config):
    # Crea una conexión a la base de datos
    conn = psycopg2.connect(
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"]
    )
    cur = conn.cursor()

    # Lee el archivo CSV
    df = pd.read_csv(csv_file, on_bad_lines='skip')

    # Concatena todos los campos de cada registro en una sola cadena
    df['datos'] = df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    data = [list(row) for row in df[['datos']].itertuples(index=False)]

    # Crea la tabla en PostgreSQL
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (datos text);")

    # Inserta los datos en la tabla
    cur.executemany(f"INSERT INTO {table_name} (datos) VALUES (%s);", data)

    # Crea un índice invertido en la tabla
    cur.execute(f"CREATE INDEX {table_name}_gin ON {table_name} USING gin(to_tsvector('english', 'datos'));")

    # Confirma los cambios y cierra la conexión
    conn.commit()
    cur.close()
    conn.close()

"""
index_csv_to_postgres('../styles.csv', 'proyecto', {
    "user": "postgres",
    "password": "1234",
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres"
})
"""

app = QApplication([])
window = MainWindow()
window.show()
sys.exit(app.exec_())