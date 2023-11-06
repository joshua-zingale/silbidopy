import struct
SHORT_LEN = 2
INT_LEN = 4
DOUBLE_LEN = 8
LONG_LEN = 8


HEADER_STR = "silbido!".encode("utf-8")
DET_VERSION = 4

# feature bit-mask - describes what has been populated and allows
# backward compatibility
    
# features produced for every point
TIME = 1
FREQ = 1 << 1
SNR = 1 << 2
PHASE = 1 << 3
    
RIDGE = 1 << 6

TIMESTAMP = 1 << 7  # base timestamp for detections
USERCOMMENT = 1 << 8  # user comment field
    
# features produced once per call
SCORE = 1 << 4
CONFIDENCE = 1 << 5
SPECIES = 1 << 9
CALL = 1 << 10

DEFAULT = TIME | FREQ

def writeTimeFrequencyBinary(filename, contours):
    '''Writes only time and frequency and leaves no comment nor timestamp.
    graphId written is currently an arbitrary number and is the same for each write.

    UserVersion is currently set to 0. I do not know what userversion means.
    
    :param filename: the name of the file to which to be written
    :param contours: A two dimensional array containing tuples (time, frequence),
                      where both are a floating point number.
                      e.g. [[(1.2,75.23), (1.25, 74.77)], [(4.9,62.48), (5.52, 60.29)]]
    '''
    
    version = DET_VERSION.to_bytes(SHORT_LEN, byteorder = "big")
    bitMask = (TIME | FREQ).to_bytes(SHORT_LEN, byteorder = "big")
    userVersion = (0).to_bytes(SHORT_LEN, byteorder = "big")
    headerSize = (3 * SHORT_LEN + INT_LEN + len(HEADER_STR)).to_bytes(INT_LEN, byteorder = "big") # ASSUMES no comments


    file = open(filename, 'wb')

    # Write magic string
    file.write(HEADER_STR)

    # Write header
    file.write(version)
    file.write(bitMask)
    file.write(userVersion)
    file.write(headerSize)


    # Write tonal meta deta
    # graphId is an arbitrary value as of now
    graphId = (14567891234567891234).to_bytes(LONG_LEN, byteorder = "big")
    

    # Write tonals  
    for contour in contours:
        N = len(contour).to_bytes(INT_LEN, byteorder = "big") # number of points in contour
        
        file.write(graphId)
        file.write(N)

        # Write all time and frequency nodes for the current contour
        for time, freq in contour:
            file.write(struct.pack('>d', time))
            file.write(struct.pack('>d', freq))