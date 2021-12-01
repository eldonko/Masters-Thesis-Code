from SGTEConverter.Development.SGTEConverter import SGTEConverter


def main():
	sgte_converter = SGTEConverter(r"C:\Users\danie\Documents\Montanuni\Masterarbeit\2 Literatur\SGTE Data For Pure Elements [Dinsdale, A.].pdf")
	sgte_converter.read_doc()


if __name__ == '__main__':
	main()