import dash
from dash import html, dcc, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import json

FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

app.config['suppress_callback_exceptions'] = True
app.scripts.config.serve_locally = True
server = app.server