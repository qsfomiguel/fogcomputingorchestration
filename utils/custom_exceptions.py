class InvalidValueError(Exception):

    def __init__(self, message, rg="[not defined here, verify documentation]"):
        super(InvalidValueError, self).__init__(message)
        self.msg = message 
        self.rg = rg 
    
    def __str__(self):
        return f'{self.msg} -> acceptable range {self.rg}'

class InvalidStateError(Exception):
    
    def __init__(self, message, st="[unknown]"):
        super(InvalidStateError, self).__init__(message)
        self.msg = message 
        self.st = st 

    def __str__(self):
        return f'{self.msg}-> object state {self.st}'
        