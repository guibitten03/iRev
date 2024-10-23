def regulate_metrics(values, lenght):
    if len(values) != lenght:

        size_of_batch = len(values) / lenght
        if type(size_of_batch) == float:
            size_of_batch = math.ceil(size_of_batch)
            pad = size_of_batch * lenght
            values = np.pad(values, (pad - len(values)))


        new_values = []
        old_i = 0
        for i in range(size_of_batch, len(values), size_of_batch):
            new_values.append(np.sum(values[old_i:i]))
            old_i = i
        

    return new_values