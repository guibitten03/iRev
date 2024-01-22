def novelty(true, pred):
  """
  Calcula a novidade de um conjunto de itens.

  Args:
    itens: Um vetor de itens.
    pontuações: Um vetor de pontuações dos itens.

  Returns:
    Um valor entre 0 e 1, sendo 0 a menor novidade e 1 a maior novidade.
  """

  # Obtém a média das pontuações.
  media = sum(pred) / len(pred)

  # Calcula a novidade de cada item.
  novidades = []
  for t in true:
    novidades.append(t - media)

  # Retorna a média das novidades.
  return abs(sum(novidades) / len(novidades))