import os


# Rename all files to fit keras format
def main():
    i = 1

    # Rename overripe bananas
    for file in os.listdir("overripe"):
        dst = "overripe." + str(i) + ".jpeg"
        src = "overripe/" + file
        dst = "overripe/" + dst

        os.rename(src, dst)
        i += 1

    i = 1

    # Rename ripe bananas
    for file in os.listdir("ripe"):
        dst = "ripe." + str(i) + ".jpeg"
        src = "ripe/" + file
        dst = "ripe/" + dst

        os.rename(src, dst)
        i += 1

    i = 1

    # Rename unripe bananas
    for file in os.listdir("unripe"):
        dst = "unripe." + str(i) + ".jpeg"
        src = "unripe/" + file
        dst = "unripe/" + dst

        os.rename(src, dst)
        i += 1


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
