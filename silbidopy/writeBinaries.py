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


def writeContoursBinary(filename, contours,
        time = True, frequency = True, snr = False,
        phase = False, ridge = False,
        comment = "", timestamp = "", score = False,
        confidence = False, species = False, call = False):
    '''Writes contours to a silbido binary file.

    UserVersion is currently set to 0. I do not know what userversion means.
    The fields that are to be present in this annotation file must be specified
    in the keyword arguments, which all, save comment and timestamp, are boolean.
    comment and timestamp being the empty string means that there will not be saved
    a comment or timestamp respectively. The graphid is set to an arbitrary constant value
    for each tonal.
    
    :param filename: the name of the file to which to be written
    :param contours: an array containing dictionaries, each of which represents on contour
        Each dictionary must contain the fields specified in the keyword arguments.
        If all fields are specified, then the dictionary would look as follows:
        {"score": 0.8, "confidence": 1.0, "species": "Baleen Whale", "call": "D",
        "tfnodes": [
            {"time": 3.25, "freq": 50.125, "snr": 6.6, "phase", 0.25, "ridge", 1.0},
            {"time":...},
            ...,
        ]}
    '''
    
    version = DET_VERSION.to_bytes(SHORT_LEN, byteorder = "big")
    bitMask = 0
    
    headerSize = 3 * SHORT_LEN + INT_LEN + len(HEADER_STR)
    
    if time:
        bitMask += TIME
    if frequency:
        bitMask += FREQ
    if snr:
        bitMask += SNR
    if phase:
        bitMask += PHASE
    if ridge:
        bitMask += RIDGE
    if len(comment) > 0 :
        bitMask += USERCOMMENT
        headerSize += 2 + len(comment)
    if len(timestamp) > 0 :
        bitMask += TIMESTAMP
        headerSize += 2 + len(timestamp)
    if score:
        bitMask += SCORE
    if confidence:
        bitMask += CONFIDENCE
    if species:
        bitMask += SPECIES
    if call:
        bitMask += CALL

    bitMask = bitMask.to_bytes(SHORT_LEN, byteorder = "big")
    userVersion = (0).to_bytes(SHORT_LEN, byteorder = "big")
    headerSize = headerSize.to_bytes(INT_LEN, byteorder = "big")


    file = open(filename, 'wb')

    # Write magic string
    file.write(HEADER_STR)

    # Write header
    file.write(version)
    file.write(bitMask)
    file.write(userVersion)
    file.write(headerSize)
    if comment:
            L = len(comment)
            if L >= 2**16:
                raise RuntimeException("Length of comment is too long")
            file.write(L.to_bytes(SHORT_LEN, byteorder="big"))
            file.write(comment.encode("utf-8"))
    if timestamp:
            L = len(timestamp)
            if L >= 2**16:
                raise RuntimeException("Length of timestamp is too long")
            file.write(L.to_bytes(SHORT_LEN, byteorder="big"))
            file.write(timestamp.encode("utf-8"))


    # Write tonal meta deta
    # graphId is an arbitrary value as of now
    graphId = (14567891234567891234).to_bytes(LONG_LEN, byteorder = "big")
    

    # Write tonals  
    for contour in contours:
        
        if confidence:
            file.write(struct.pack(">d", contour["confidence"]))
        if score:
            file.write(struct.pack(">d", contour["score"]))
        if species:
            L = len(contour["species"])
            if L >= 2**16:
                raise RuntimeException("Length of species name is too long")

            file.write(L.to_bytes(SHORT_LEN, byteorder="big"))
            file.write(contour["species"].encode("utf-8"))
        if call:
            L = len(contour["call"])
            if L >= 2**16:
                raise RuntimeException("Length of call name is too long")

            file.write(L.to_bytes(SHORT_LEN, byteorder="big"))
            file.write(contour["call"].encode("utf-8"))
        
        file.write(graphId)

        N = len(contour["tfnodes"]).to_bytes(INT_LEN, byteorder = "big") # number of points in contour
        
        file.write(N)

        # Write all time and frequency nodes for the current contour
        for node in contour["tfnodes"]:
            if time:
                file.write(struct.pack('>d', node["time"]))
            if frequency:
                file.write(struct.pack('>d', node["freq"]))
            if snr:
                file.write(struct.pack(">d", node["snr"]))
            if phase:
                file.write(struct.pack(">d", node["phase"]))
            if ridge:
                file.write(struct.pack(">d", node["ridge"]))



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
