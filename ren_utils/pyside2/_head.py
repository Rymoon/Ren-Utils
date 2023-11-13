import sys
from pathlib import Path
import os
from PySide2.QtCore import Property, QObject, Qt, QUrl, Signal, Slot
from PySide2.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QUrl
from PySide2.QtWidgets import QApplication
from PySide2.QtGui import QGuiApplication, QImage, QPixmap
from PySide2.QtQml import QQmlApplicationEngine,qmlRegisterType
from PySide2.QtSql import QSqlDatabase, QSqlQuery, QSqlQueryModel
from dataclasses import dataclass
from enum import Enum

from typing import List, Any, Dict
import sqlite3

