class UntrainedModelException(Exception):

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None
        
    def __str__(self):
        if self.message:
            return f"UntrainedModelException: {self.message}"
        else:
            return "UntraineModelException: Model not fitted before calling transform(). Please .fit() the model before calling this method."