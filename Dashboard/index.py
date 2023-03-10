from app import *

import graphics
import tables

    
app.layout = dbc.Row([
    
        dcc.Location(id = "url"),
        
        dbc.Col([
        
            dbc.Row([
                dbc.Col([ # Table Sidebar
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([ # Title
                                    html.H3("Tables")
                                ], className = "text-center")
                            ]),
                            
                            dbc.Row([ # Default Hyper Parameters Button
                                dbc.Button("Default Hyper Parameters", color = "primary", className = "me-1")
                            ])
                            
                        ])
                    ], className = "p-5 h-50"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H3("Pages"),
                                    
                                ], className = "text-center")
                            ]),
                            
                            dbc.Nav([
                               dbc.NavLink("Graphics", href = "graphics",id = "graphs-btn", active = True, className="mb-3"), 
                               dbc.NavLink("Tables", href = "tables", id = "table-btn", active = True), 
                            ], vertical=True, pills=True, fill = True, id='nav_buttons', className = "text-center"),
                            
                        ])
                    ], className = "p-5 h-50")
                    
                ], md = 2, className = "p-3 h-100"),
                
                dbc.Col([ # Content
                    
                html.Div(id = "page-content")
                    
                    
                ], md = 10, className = "p-3")
            ], className = "p-3 h-100")
        
    ])
])


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def pagenation(path):
    if path == "/" or path == "/graphics":
        return graphics.layout
    
    if path == "/tables":
        return tables.layout


if __name__ == "__main__":
    app.run_server(debug = True)
