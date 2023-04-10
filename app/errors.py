class ParameterRequiredException(BaseException):
    def __init__(self, parameter_name):
        alg, parameter = parameter_name.split("_")
        self.message = f"Missing required parameter '{parameter}' for algorithm {alg}"
        super().__init__(self.message)
