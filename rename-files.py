import os


# Rename all files to fit keras format
def main():
    i = 1

    # Rename overripe bananas
    for file in os.listdir("bananas/overripe"):
        dst = "overripe." + str(i) + ".jpeg"
        src = "bananas/overripe/" + file
        dst = "bananas/overripe/" + dst

        os.rename(src, dst)
        i += 1

    i = 1

    # Rename ripe bananas
    for file in os.listdir("bananas/ripe"):
        dst = "ripe." + str(i) + ".jpeg"
        src = "bananas/ripe/" + file
        dst = "bananas/ripe/" + dst

        os.rename(src, dst)
        i += 1

    i = 1

    # Rename unripe bananas
    for file in os.listdir("bananas/unripe"):
        dst = "unripe." + str(i) + ".jpeg"
        src = "bananas/unripe/" + file
        dst = "bananas/unripe/" + dst

        os.rename(src, dst)
        i += 1


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
