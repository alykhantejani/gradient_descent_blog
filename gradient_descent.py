def run(data_points, initialize_params, calculate_derivs, calculate_error, max_steps = 30, learning_rate = 0.1):
	params = initialize_params()
	error = calculate_error(params)

	all_parameters = []
	all_parameters.append((list(params), error))

	for step in range(1, max_steps):
		derivs = calculate_derivs(data_points, params)

		for i in range(0, len(params)):
			params[i] = params[i] - learning_rate * derivs[i]

		error = calculate_error(params)

		all_parameters.append((list(params), error))

	return all_parameters