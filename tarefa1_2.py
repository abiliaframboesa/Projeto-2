import json
import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image # python package Pillow


with open('dados/prize.json','r') as f:
    prizes = json.load(f)["prizes"]

with open('dados/laureate.json','r') as f:
    laureates = json.load(f)["laureates"]

print("Tarefa 1")
print("Exercício 1")


def maisPartilhados() -> tuple[int, set[tuple[int, str]]]:
    """ retorna o número máximo de co-laureados e um conjunto de pares (ano,categoria)
     com os premios atribuidos ao maior numero de co-laureados"""

    #percorrer a lista de premios até chegar a cada laureado
    #criar variavel para verificar logo o primeiro valor de share 
    #se esse valor de share for maior que o da variavel max_share inicialmente definida entao o valor dessa variavel é atualizado e redefine o conjunto de prémios
    #no caso de o valor de share ser igual ao da variavel, o premio correspondente é adicionado ao set

    max_share = 0
    max_share_prizes = set()
   
    for prize in prizes:
        year = prize["year"]
        category = prize["category"]
        if "laureates" in prize:
            laureates = prize["laureates"]
            if laureates:  
                first_max_share = int(laureates[0]["share"]) #variavel que procura pelo primeiro valor maximo de share 
                if first_max_share > max_share:
                    max_share = first_max_share
                    max_share_prizes = {(year, category)} #redefine o conjunto de prémios
                elif first_max_share == max_share:
                    max_share_prizes.add((year, category))

    return max_share, max_share_prizes

partilhados = maisPartilhados()
print(partilhados)

# mais = (4, {('1981', 'physics'), ('1958', 'medicine'), ('2001', 'chemistry'), ('1973', 'physics'), ('2002', 'physics'), ('2011', 'medicine'), ('2015', 'medicine'), ('1997', 'chemistry'), ('1977', 'medicine'), ('2021', 'physics'), ('1947', 'medicine'), ('2002', 'chemistry'), ('2000', 'physics')})
# print(mais==partilhados)

print()
print("Exercício 2")

def multiLaureados() -> dict[str,set[str]]:
    """Complete a definição da função multiLaureados, que retorna um dicionário com as 
    personalidades laureadas em mais do que uma categoria."""

    multi={}
    for prize in prizes:
        if "laureates" in prize:
            laureates=prize["laureates"]
            for laureate in laureates:
                if "surname" in laureate:
                    nome = laureate["firstname"] + " " + laureate["surname"]
                else:
                    nome = laureate["firstname"]
            if nome in multi:
                multi[nome].add(prize["category"]) #adiciona uma nova categoria ao dicionario
                #print(multi)
            else:
                 multi[nome] = {prize["category"]} #o correspondente à chave nome é o dicionário das categorias
                 #print(multi)
    
    multi = {nome: categorias for nome, categorias in multi.items() if len(categorias) > 1} #len>1 para dizer que recebeu mais que um premio

    return multi

multi=multiLaureados()
print(multi)

# certo={'Linus Pauling': {'chemistry', 'peace'}, 'Marie Curie': {'chemistry', 'physics'}}
# print(multi==certo)

print()
print("Exercício 3")

def anosSemPremio() -> tuple[int, int]:
    anos_sem_premio = set()
    awarded_years = set()

    # Adiciona todos os anos em que prêmios foram concedidos ao conjunto awarded_years
    for prize in prizes:
        if "year" in prize:
            year=prize["year"]
            awarded_years.add(int(year))
        if "laureates" not in prize or not prize["laureates"]: #Verificar se há "laureates" no prize ou se a condição prize["laureates"] é falsa --> significa que não há laureados para este prémio
            anos_sem_premio.add(int(int(year)))

    # Encontra o intervalo mais longo de anos consecutivos sem prémios concedidos
    consecutive_years = 0
    max_consecutive_years = 0
    start_year = 0
    end_year = 0

    for year in sorted(list(awarded_years)):
        if year in anos_sem_premio:
            consecutive_years += 1
            if consecutive_years > max_consecutive_years:
                max_consecutive_years = consecutive_years
                start_year = year - consecutive_years + 1
                end_year = year
        else:
            consecutive_years = 0

    return start_year, end_year

anos=anosSemPremio()
print(anos)

print()
print("Exercício 4")


def rankingDecadas() -> dict[str,tuple[str,int]]:
    """Complete a definição da função rankingDecadas que retorna, por cada década, o país com mais laureados, 
    considerando o país de afiliação de cada laureado. Nota: Leia os dados do ficheiro laureate.json.
    """
    rank={}
    for laureate in laureates:
         if "prizes" in laureate:
              prizes=laureate["prizes"]
              for prize in prizes:  
                if "year" in prize:
                    year=int(prize["year"])
                    decade = str((year // 10) * 10)  # calcular a década
                    decade = decade[:-1] + 'x' # strings são imutáveis --> não consigo atualizar o valor da string. neste caso criei uma nova com as alterações que queria fazer                     
                    
                    if decade not in rank:
                        rank[decade] = {}  # Inicía o dicionário para a década = colocar a década no dicionário se ela mão estiver lá
                    #else:
                        #rank[decade] = rank[decade]
                    if "affiliations" in prize:
                        affiliations=prize["affiliations"]
                        for affiliation in affiliations:
                            if "country" in affiliation:
                                country=affiliation["country"]

                                # country é visto como uma key do dicionário 
                                # 1 é o valor
                                dic_countries=rank[decade] # dic_countries=rank[decade] é um novo dicionário 
                                if country not in dic_countries:
                                    dic_countries[country] = 1 # inicía a contagem de laureados para um país que ainda não está na lista para a década atual
                                else:
                                    dic_countries[country] += 1

                        
    for decade, dic_countries in rank.items():    # Encontrar o país com mais laureados na década
         #key=decade; valor=country
         # country é dicionário que assume:
         # key = most_laureated_country;  valor = num_laureates
         most_laureated_country, num_laureates = max(dic_countries.items(), key=lambda x: x[1])
         rank[decade] = (most_laureated_country, num_laureates)

    return rank

rank=rankingDecadas()
print(rank)

# rank1 = {'190x': ('Germany', 11), '191x': ('Germany', 9), '192x': ('Germany', 8), '193x': ('Germany', 16), '194x': ('USA', 15), '195x': ('USA', 33), '197x': ('USA', 45), '196x': ('USA', 30), '198x': ('USA', 47), '199x': ('USA', 58), '200x': ('USA', 79), '202x': ('USA', 29), '201x': ('USA', 75)}
# print(rank==rank1)


print()
print("Tarefa 2")
print("Exercício 1")

def toGrayscale(rgb:np.ndarray) -> np.ndarray:
    """Converta uma imagem a cores (em que uma cor RGB é definida por três componentes do tipo uint8 entre 0 e 255), 
    numa imagem em tons de cinzento (em que uma cor Grayscale é definida por um componente do tipo uint8 entre 0 e 255). 
    Existem diferentes formas de converter uma cor RGB numa cor Grayscale. 
    Uma das formas é calcular a luminância da cor, dada pela seguinte fórmula y=0.21*r+0.72*g + 0.07*b
    (os humanos são mais sensíveis à cor verde daí esta ter mais peso na média pesada): 
    . Complete a definição da função toGrayscale que recebe um array correspondente a uma imagem a cores 
    e retorna um array correspondente a uma imagem em tons de cinzento."""

    #y=0.21*r + 0.72*g + 0.07*b ---formula rgb
    
    # Calcular a luminescencia de cada pixel do array rgb (0-red, 1-green, 2-blue)
    luminance = 0.21 * rgb[..., 0] + 0.72 * rgb[..., 1] + 0.07 * rgb[..., 2]

    grayscale_image = luminance.astype(np.uint8) # converter valores de luminescencia para uint8 (0-255)
    
    return grayscale_image


color_image = np.asarray(Image.open("Projeto2/dados/test.png"))
grayscale_image = toGrayscale(color_image) #converte array a cores para array escala cinzentos

grayscale_pil_image = Image.fromarray(grayscale_image) #cria imagem
grayscale_pil_image.show()

expected_gray = np.array(Image.open("Projeto2/dados/test_gray.png"))
#print(np.testing.assert_array_equal(grayscale_image, expected_gray))



print()
print("Exercício 2")

def toBW(gray:np.ndarray,threshold:tuple[int,int]) -> np.ndarray:
    """Converta uma imagem em tons de cinzento numa imagem a preto e branco (em que cada cor é dada pelo valor 0 para preto 
    ou 255 para branco). Complete a definição da função toBW que lê uma imagem em tons de cinzento e escreve numa imagem a preto e branco. 
    Esta função recebe também um threshold de cores que serão convertidas em branco, na forma de um intervalo fechado entre 0 e 255. 
    As restantes cores serão convertidas em preto."""

    #criar um array de gray com zeros para colocar os valores de bw
    #definir o threshold para branco (valor 255)
    #iterar sobre cada pixel no array gray e ver se está dentro do threshold para branco
    #se estiver valor do pixel é 255 = branco; se não estiver valor do pixel é 0 = preto
    
    black_white = np.zeros_like(gray, dtype=np.uint8) #array de zeros com estrutura semelhante à de gray
    lower, upper = threshold
    
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            pixel_value = gray[i, j]  
            
            if lower <= pixel_value <= upper:
                black_white[i, j] = 255  # Mudar valor do pixel para branco
            else:
                black_white[i, j] = 0  # Mudar valor do pixel para preto
    
    return black_white

color_image = np.asarray(Image.open("Projeto2/dados/test.png"))
test_bw = np.asarray(Image.open("Projeto2/dados/test_bw.png"))

grayscale_image = toGrayscale(color_image)

bw_image = toBW(grayscale_image, (0, 20))
bw_image = Image.fromarray(bw_image) #cria imagem
bw_image.show()

#print(np.testing.assert_array_equal(bw_image, test_bw))


print()
print("Exercício 3")

def autoThreshold(fromimg:str,tolerance:int) -> tuple[int,int]:
    """Calcule um threshold. Para isso, vamos assumir que uma imagem em tons de cinzento 
    tem uma cor mais preponderante correspondente ao seu fundo, que ficará a branco. 
    Complete a definição da função autoThreshold, que recebe uma tolerância i
    e lê uma imagem em tons de cinzento e returna um threshold, ou seja, um intervalo [c-1, c+1]
    de cinzentos (entre 0 e 255) correspondente ao fundo, centrado na cor mais frequente 
    ."""

    #contar frequencia de cores
    #ver qual o tom de cinzento mais frequente --> corresponde ao fundo
    #calcular o threshold inferior e o superior sabendo que têm de ser valores entre 0 e 255

    grayscale: np.ndarray = asarray(Image.open(fromimg)) #array em escala de cinzentos
    
    freq_colors = np.bincount(grayscale.flatten(), minlength=256) #contar frequencia de cores --> cada elemento corresponde a uma cor da escala e o valor correspondente a esse elemento é a frequencia dessa cor
    #print(freq_colors) 

    most_common_color = np.argmax(freq_colors) #argmax diz o indice da cor mais frequente do array
    #print(most_common_color)
    
    #o threshold tem de estar obrigatoriamente entre 0 e 255 por isso é que se definem o max e min
    lower = max(0, most_common_color - tolerance)
    upper = min(255, most_common_color + tolerance)

    return lower, upper 

threshold = autoThreshold("Projeto2/dados/test.png", tolerance=5)
print("Threshold:", threshold)


print()
print("Exercício 4")

def toContour(bw:np.ndarray) -> np.ndarray:
    """Converta uma imagem a preto e branco, com fundo branco, numa imagem a preto e branco do seu contorno. 
    Complete a definição da função toContour que, para cada pixel da imagem inicial, o coloca a preto na imagem 
    resultante caso algum dos seus píxeis circundantes tiver uma cor diferente, e a branco vice-versa. 
    Nota: Outros algoritmos para calcular contornos serão aceites, desde que façam sentido."""

    # iterar sobre cada pixel, exceto os das bordas (tanto linhas como colunas)
    # se algum dos pixeis ao lado, em cima e em baixo for diferente coloca preto (0)
    # else coloca branco (255)

    contour = np.zeros_like(bw, dtype=np.uint8)
    
    # Iterar sobre cada pixel, exceto as bordas
    for i in range(1, bw.shape[0] - 1):
        for j in range(1, bw.shape[1] - 1):
            if (bw[i, j] != bw[i - 1, j] or bw[i, j] != bw[i + 1, j] or bw[i, j] != bw[i, j + 1] or bw[i, j] != bw[i, j - 1]):
                contour[i, j] = 0
            else:
                contour[i, j] = 255
    return contour

bw_image = np.asarray(Image.open("Projeto2/dados/test_bw.png"))
contour_image = toContour(bw_image)

contour_image = Image.fromarray(contour_image) #cria imagem
contour_image.show()

# print(np.testing.assert_array_almost_equal(toContour(bw_image),contour_image,decimal=0))


