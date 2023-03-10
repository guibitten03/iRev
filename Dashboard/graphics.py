from app import *

layout = html.Div([
    
                dcc.Store("ex_setup"),
                html.Div(id = "default-input-callback"),
                
                dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H3("Datasets")
                                        ]),
                                        
                                        dbc.Col([
                                            dbc.Button("Amazon", color = "primary", className = "me-1 col-6", id = "amz-btn")
                                            
                                        ], align = "center"),
                                        dbc.Col([
                                            dbc.Button("Yelp", color = "primary", className = "me-1 col-6", id = "yel-btn")

                                        ], align = "center")
                                        
                                    ]),
                                    
                                ])
                            ], className = "p-3"),
                            
                        ]),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.H3("Sizes")
                                        ]),
                                        
                                        dbc.Col([
                                            dbc.Button("@10", color = "primary", className = "me-1 col-6", id = "list-size-10"),
                                            
                                        ], align = "center"),
                                        dbc.Col([
                                            dbc.Button("@20", color = "primary", className = "me-1 col-6", id = "list-size-20"),

                                        ], align = "center")
                                        
                                    ]),
                                ])
                            ], className = "p-3")
                        ])
                    ]),
    
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                                dbc.Col([
                                    html.H5("Algorithms Selection: "),
                                    dcc.Dropdown(
                                        placeholder="Select Algorithms",
                                        id = "alg-drop",
                                        multi = True
                                    )
                                ])
                            ])
                        ])
                ], className = "p-3 mt-3"),
                    
                dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody(dcc.Graph(id = "mse-fig"))
                                    ], className="p-3")
                                ]),
                                
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody(dcc.Graph(id = "rmse-fig"))
                                    ], className="p-3")
                                ]),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody(dcc.Graph(id = "mae-fig"))
                                    ], className="p-3")
                                ]),
                            ]),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody(dcc.Graph(id = "diversity-fig"))
                                    ], className="p-3")
                                ]),
                            ])
                            
                        ], className = "p-3", md = 6),
                        
                        dbc.Col([
                            dbc.Row([
                                # Razão entre mse e mae
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody(dcc.Graph(id = "msexmae-fig"))  
                                    ], className="p-3")
                                ])
                            ]),
                            
                            # Media de erros sobre o tempo
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody(dcc.Graph(id = "errorxtime-fig"))
                                    ],  className="p-3")
                                ])
                            ])
                        ], md = 6, className="p-3")
                ])
        ])

@app.callback(
    Output("ex_setup", "data"),
    Input("default-input-callback", "children")
)
def charge_dataframe(none_input):
    return pd.read_csv("Experimental_setup.csv").to_dict()


@app.callback(
    [
        Output("alg-drop", "options"),
        Output("alg-drop", "value")
    ],
    Input("ex_setup", "data")
)
def charge_dropdown_options(data):
    df = pd.DataFrame(data)
    algorithms = df['Algorithm'].drop_duplicates(keep = 'first').values
    return {i:i for i in algorithms}, algorithms


@app.callback(
    [
        Output("mse-fig", "figure"),
        Output("rmse-fig", "figure"),
        Output("mae-fig", "figure"),
        Output("diversity-fig", "figure"),
        Output("msexmae-fig", "figure"),
        Output("errorxtime-fig", "figure"),
    ],
    
    [
        Input("amz-btn", "n_clicks"),
        Input("yel-btn", "n_clicks"),
        Input("ex_setup", "data"),
        Input("alg-drop", "value"),
        Input("list-size-10", "n_clicks"),
        Input("list-size-20", "n_clicks"),
    ],
    
)
def charge_figures(n, n_y, data, values, ten, twe):
    df = pd.DataFrame(data)
    
    dataset = "Yelp" if ctx.triggered_id == "yel-btn" else "Amazon_Music" 
    
    size = 10 if ctx.triggered_id == "list-size-10" else 20
    
    df = df.loc[df['Dataset'] == dataset]
    df = df.loc[df['Algorithm'].isin(values)]
    
    mse_figure = px.bar(df, x = 'Algorithm', y = 'mse', title = f"{dataset} dataset", text = 'mse', template="seaborn")
    rmse_figure = px.bar(df, x = 'Algorithm', y = 'rmse', title = f"{dataset} dataset", text = 'rmse', template="seaborn")
    mae_figure = px.bar(df, x = 'Algorithm', y = 'mae', title = f"{dataset} dataset", text = 'mae', template="seaborn")
    diversity_figure = px.bar(template="seaborn")
    
    # --- #
    
    mse_mae = df[['Algorithm', 'mse', 'mae']]
    mse_mae['Relação entre MSE e MAE'] = round(df['mse'] * df['mae'], 2)
    
    msexmae = px.bar(mse_mae, x = 'Algorithm', y = 'Relação entre MSE e MAE', title = "Relação entre MSE e MAE", text = "Relação entre MSE e MAE", template="seaborn")
    
    error_time = df[['Algorithm', 'mse', 'mae', 'time']]
    error_time['error_time'] = round((((df['mse'] + df['mae']) / df['time']) * 100000), 5)
    
    error_time = px.bar(error_time, x = 'Algorithm', y = 'error_time', title = "Relação entre Erro e Tempo", text = "error_time", template="seaborn")
        
    return mse_figure, rmse_figure, mae_figure, diversity_figure, msexmae, error_time