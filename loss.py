def loss(intended, received):
    intended = [int(0 == intended), int(1 == intended), int(2 == intended), int(3 == intended), int(4 == intended),
                int(5 == intended), int(6 == intended), int(7 == intended), int(8 == intended), int(9 == intended)]
    
    for i in intended:
        error += (intended[i] - received[i]) ** 2
    
    return error * 0.1

def error(intended, received):
    intended = [int(0 == intended), int(1 == intended), int(2 == intended), int(3 == intended), int(4 == intended),
                int(5 == intended), int(6 == intended), int(7 == intended), int(8 == intended), int(9 == intended)]
    
    error = [0] * 10

    for i in intended:
        error[i] = 2 * (received[i] - intended[i]) / 10
    
    return error
