Pateikta programa naudoja populiarų duomenų rinkinį FashionMNIST, kad apmokyti neuroninį tinklą klasifikuoti vaizdus. Programoje įvykdytas duomenų paruošimas, apdorojimas, pritaikomas modelis ir stebimi rezultatai.

## **Duomenų paruošimas**

* Duomenys atsitiktinai padalijami į tris aibes: mokymo, validavimo ir testavimo.

* 10 % duomenų atsitiktinai parenkama testavimui ir validavimui. Likusi dalis naudojama mokymui.

## **Duomenų apdorojimas:**

* Duomenys normalizuojami.

* Vaizdai perdaromi į 28x28 matricą.

* Tikslinės reikšmės užkoduojamos naudojant „one-hot“ metodą.

## **Modelio architektūra:**

* Modelis susideda iš dviejų konvoliucinių sluoksnių, normalizavimo sluoksnių, dropout sluoksnių (reguliavimui), sluoksnio, kuris paverčia dvimatį masyvą į vienmatį, ir pilnai sujungto sluoksnio.

* Paskutinis išvesties sluoksnis naudoja softmax aktyvacijos funkciją, kad klasifikuotų į 10 kategorijų.

## **Modelio mokymas**

* Modelis naudoja „Adam“ optimizatorių ir kategorijų kryžminės entropijos nuostolius.

* Modelis mokomas 10 epochų naudojant mokymo duomenis ir validuojant su validavimo duomenimis.

## **Modelio įvertinimas:**

* Po mokymo modelis įvertinamas testavimo aibei apskaičiuojant tikslumą.

* Pavaizduojama klasifikavimo matrica, kuri rodo modelio klasifikavimo rezultatus.

* Pavaizduojami grafikai, rodantys mokymo/validavimo paklaidas ir tikslumą skirtingose epochose.
