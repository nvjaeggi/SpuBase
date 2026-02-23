import os
import numpy as np
import pandas as pd


def normalize_dfs(df_a):
    for dd, df in enumerate(df_a):
        df.dropna(inplace=True, axis=1)
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.T
        df /= df.sum(axis=0).values[0]
        df_a[dd] = df
    return df_a


def surface_composition(self, comp_df=None, form=None):
    """
    Determine atomic composition based on mineral composition
    """

    if comp_df is not None:
        # Cleans up passed composition dataframe by dropping zero columns
        # and any mineral that is not part of the databse.
        comp_df = comp_df.loc[:, (comp_df != 0).any(axis=0)]
        mineral_all = [col for col in comp_df.T.columns]
        comp_df.drop(self.min_not_included, inplace=True, errors='ignore', axis=1)

        if form == 'mol%':
            minfrac_mol = comp_df
            minfrac_wt = comp_df.mul(pd.Series(self.amu_minerals_dic))
            minfrac_vol = minfrac_wt.div(pd.Series(self.wt_density_df_minerals_dic))

        elif form == 'wt%':
            minfrac_wt = comp_df
            minfrac_vol = comp_df.div(pd.Series(self.amu_minerals_dic))
            minfrac_mol = comp_df.div(pd.Series(self.wt_density_df_minerals_dic))

        elif form == 'vol%':
            minfrac_vol = comp_df
            minfrac_wt = comp_df.mul(pd.Series(self.wt_density_df_minerals_dic))
            minfrac_mol = minfrac_wt.div(pd.Series(self.amu_minerals_dic))

        else:
            print('Volume percent mineral fractions were assumed.\n'
                  'Pass form="mol%" or "wt%" instead for molar/weight fractions\n')
            minfrac_vol = comp_df
            minfrac_wt = comp_df.mul(pd.Series(self.wt_density_df_minerals_dic))
            minfrac_mol = minfrac_wt.div(pd.Series(self.amu_minerals_dic))

        minfrac_wt, minfrac_vol, minfrac_mol = normalize_dfs([minfrac_wt, minfrac_vol, minfrac_mol])

        self.minfrac_df_volume = minfrac_vol
        self.minfrac_df_volume.drop(self.min_not_included, inplace=True, errors='ignore', axis=1)

    else:
        self.minfrac_df_volume = self.minfrac_df_volume.loc[:, (self.minfrac_df_volume != 0).any(axis=0)]
        mineral_all = [col for col in self.minfrac_df_volume.T.columns]
        self.minfrac_df_volume.drop(self.min_not_included, inplace=True, errors='ignore', axis=0)

    # Creates separate array for mineral names and mineral fractions
    try:
        self.mineral_a = [col for col in self.minfrac_df_volume.T.columns]

    except:
        self.mineral_a = [col for col in self.minfrac_df_volume.T.columns]

    self.minfrac_a = self.minfrac_df_volume.T.values.tolist()[0]

    # Get mineral dictionary that connects elements to minerals
    mindict_df_atoms = self.mindict_df_atoms.loc[:, self.mindict_df_atoms.columns != 'total']
    out = []
    for mm, mineral in enumerate(self.mineral_a):
        out.append(mindict_df_atoms.loc[mineral].mul(self.minfrac_a[mm]).values.tolist())
    mincomp_df = pd.DataFrame(out, columns=list(mindict_df_atoms), index=self.mineral_a)
    sumcomp_df = mincomp_df.sum()
    sumcomp_df = sumcomp_df[~(sumcomp_df == 0)]  # drop all rows that are zero

    # obtain all non-zero elements that are present
    self.species_a = sumcomp_df.index.values.tolist()

    # Print mineral composition and fraction of total mineralogy present within database
    print(f'\nSum of mineral data available in SpuBase is {sumcomp_df.sum():0.2f}/1.00\n')
    if sumcomp_df.sum() < 0.99:
        missing_minerals = list(set(mineral_all).intersection(self.min_not_included))
        print(f'Minerals not in SpuBase: {missing_minerals}')


def cipw_norm(self, at_l, at_frac_l, verbose_cipw=False):
    # Adapted from "Calculating of a CIPW norm from a bulk chemical analysis"
    # by Kurt Hollocher,
    # Geology Department
    # Union College
    # Schenectady, NY 12308
    # U.S.A

    if self.isDebug:
        verbose_cipw = True

    amu_min_dict = self.amu_minerals_dic
    rho_min_dict = self.wt_density_df_minerals_dic
    mindict_df_atoms = self.mindict_df_atoms.iloc[:, self.mindict_df_atoms.columns != 'total']
    mineral_names = mindict_df_atoms.index.tolist()
    minfrac_mol = pd.DataFrame(data=np.zeros(len(mineral_names)), columns=['frac'],
                               index=mineral_names).T

    comp_df = pd.DataFrame(data=at_frac_l, columns=['at%'], index=at_l)

    species_dict_dir = os.path.join(self.tabledir, 'species_dict.txt')
    species_df = pd.read_csv(species_dict_dir, header=0, delim_whitespace=True)
    species_dict = {species_df['metal'].values[i]: species_df['species'].values[i] for i in
                    range(len(species_df['species'].values))}
    species_cat_dict = {species_df['metal'].values[i]: species_df['cation'].values[i] for i in
                        range(len(species_df['species'].values))}

    """
    Check if  composition was passed as elements or as oxides
    """
    cipw_elements = False
    element_l = ['O', 'Si', 'Ca', 'Mg']  # elements that should be passed as oxides
    el_test = [i for i in element_l if i in at_l]
    if len(el_test) > 0:
        cipw_elements = True
        cipw_oxides = not cipw_elements
        print(f'Composition given as oxides: {cipw_oxides}')

    """
    2 divide oxide weights by their respective formula weights to give molar oxide proportions
    """
    if cipw_elements:
        if comp_df.loc['O'].values <= 0.47:
            comp_df = comp_df.T.div(pd.Series(self.amu_dic))
            comp_df = comp_df.T
            print('Composition with < 47 % oxygen - wt% assumed')
        elif comp_df.loc['O'].values == 0:
            comp_df = comp_df.T.div(pd.Series(self.amu_dic))
            comp_df = comp_df.T
            print('Composition without oxygen - wt% assumed')
        else:
            print('Composition with > 47 % oxygen - at% assumed')
        comp_df = comp_df.T.div(pd.Series(species_cat_dict))
        comp_df.columns = comp_df.columns.map(species_dict.get)

    else:
        comp_df = comp_df.T.div(pd.Series(self.amu_oxides_dic))

    comp_df = comp_df / comp_df.sum(axis=1).iloc[0]

    comp_df.dropna(inplace=True, axis=1)  # drop all NAN columns

    if verbose_cipw:
        print(comp_df)

    """
    ** Sulfides: Add Mn, Cr, Ti, and Fe to Sulfur  
    """
    sulfur_limit = 1e-5
    if verbose_cipw:
        print(f'#0\nS in columns? {"S" in comp_df.columns.values.tolist()}')
    if 'S' in comp_df.columns:
        if sulfur_limit >= comp_df['S'].tolist()[0] > 0:
            print(f'Low amount of S ({comp_df["S"].values[0]:0.2e} < {sulfur_limit}) omitted')

        elif comp_df['S'].tolist()[0] > sulfur_limit:
            print(f'Transition metals are attributed to Sulfur ({comp_df["S"].values[0]:0.2e})')
            if verbose_cipw:
                print(f'#1\nCrO in columns? {"CrO" in comp_df.columns.values.tolist()}')
                print(f'Cr2O3 in columns? {"Cr2O3" in comp_df.columns.values.tolist()}')

            if 'CrO' in comp_df.columns.values.tolist():
                if 2 * comp_df['FeO'].tolist()[0] >= comp_df['CrO'].tolist()[0]:
                    minfrac_mol['Dbr'] = comp_df['CrO'].tolist()[0] / 2  # Dbr = FeCr2S4; occurance: Fe-meteorite
                    minfrac_mol['Chr'] = minfrac_mol['Dbr'].tolist()[0]  # Chr = FeCr2O4; occurance: Moon
                    comp_df['FeO'] = comp_df['FeO'] - comp_df['CrO'] / 2
                    comp_df['CrO'] = 0.00
                else:
                    minfrac_mol['Bzn'] = comp_df['CrO'].tolist()[0] / 3  # Bzn = Cr3S4; occurance: meteorites
                    comp_df['CrO'] = 0.00
            elif 'Cr2O3' in comp_df.columns.values.tolist():
                if comp_df['FeO'].tolist()[0] >= comp_df['Cr2O3'].tolist()[0]:
                    minfrac_mol['Dbr'] = comp_df['Cr2O3'].tolist()[0]  # Dbr = FeCr2S4; occurance: Fe-meteorite
                    minfrac_mol['Chr'] = minfrac_mol['Dbr'].tolist()[0]  # Chr = FeCr2O4; occurance: Moon
                    comp_df['FeO'] = comp_df['FeO'] - comp_df['Cr2O3']
                    comp_df['Cr2O3'] = 0.00
                else:
                    minfrac_mol['Bzn'] = comp_df['Cr2O3'].tolist()[0] * 2 / 3  # Bzn = Cr3S4
                    comp_df['Cr2O3'] = 0.00
            else:
                minfrac_mol['Bzn'] = 0.00
                minfrac_mol['Dbr'] = 0.00

            """
            ** Check if there is enough S to put Cr into sulfides, otherwise, keep only chromite
            """
            pS1 = comp_df['S'].tolist()[0] - \
                minfrac_mol['Bzn'].tolist()[0] * 3 - \
                minfrac_mol['Dbr'].tolist()[0] * 4

            if pS1 > 0:
                minfrac_mol['Chr'] = 0
                comp_df['S'] = pS1
            else:
                minfrac_mol['Bzn'] = 0.00
                minfrac_mol['Dbr'] = 0.00
                if verbose_cipw:
                    print(f'Not enough S to accomodate all Cr, accomodated into chromite (Chr) instead ')

            if verbose_cipw:
                print(f'remaining sulfur: {comp_df["S"].tolist()[0]}')
                print(f'remaining FeO: {comp_df["FeO"].tolist()[0]}')

            minerals = ['Abd', 'Tro', 'Was']
            oxides = ['MnO', 'FeO', 'TiO2']

            for mm, mineral in enumerate(minerals):
                oxide = oxides[mm]
                if comp_df['S'].tolist()[0] > 0:
                    if comp_df['S'].tolist()[0] >= comp_df[oxide].tolist()[0]:
                        minfrac_mol[mineral] = comp_df[oxide].tolist()[0]
                        comp_df[oxide] = 0.00
                        comp_df['S'] = comp_df['S'].tolist()[0] - minfrac_mol[mineral].tolist()[0]
                    else:
                        minfrac_mol[mineral] = comp_df['S'].tolist()[0]
                        comp_df['S'] = 0.00
                        comp_df[oxide] = comp_df[oxide].tolist()[0] - minfrac_mol[mineral].tolist()[0]

            if verbose_cipw:
                print(f'remaining sulfur: {comp_df["S"].tolist()[0]} put into 1/4 Old and 3/4 Nng')
            if comp_df['S'].tolist()[0] > 0:
                if comp_df['S'].tolist()[0] < min(comp_df['MgO'].tolist()[0] / 3 * 4,
                                                  comp_df['CaO'].tolist()[0] / 4):
                    minfrac_mol['Nng'] = comp_df['S'].tolist()[0] / 3 * 4
                    minfrac_mol['Old'] = comp_df['S'].tolist()[0] / 4
                    comp_df['MgO'] = comp_df['MgO'].tolist()[0] - minfrac_mol['Nng'].tolist()[0]
                    comp_df['CaO'] = comp_df['CaO'].tolist()[0] - minfrac_mol['Old'].tolist()[0]
                    comp_df['S'] = 0.00
    """
    3 Add MnO to FeO. 
    """

    if verbose_cipw:
        print(f'#3\nMnO in columns? {"MnO" in comp_df.columns.values.tolist()}')
    if 'MnO' in comp_df.columns:
        comp_df['FeO'] = comp_df['FeO'].tolist()[0] + comp_df['MnO'].tolist()[0]

    """
    4 Apatite: Multiply P2O5 by 3.33 and subtract this number from CaO.
    """
    if verbose_cipw:
        print(f'#4\nP2O5 in columns? {"P2O5" in comp_df.columns.values.tolist()}')
    if 'P2O5' in comp_df.columns.values.tolist():
        minfrac_mol['Ap'] = comp_df['P2O5'].tolist()[0] * 2 / 3
        comp_df['CaO'] = comp_df['CaO'] - comp_df['P2O5'] * 3.33
        comp_df['P2O5'] = 0.00

    """
    5 Ilmenite: Subtract TiO2 from FeO. Put the TiO2 value in Ilmenite
    """
    if verbose_cipw:
        print(f'#5\nTiO2 in columns? {"TiO2" in comp_df.columns.values.tolist()}')
    if 'TiO2' in comp_df.columns.values.tolist():
        minfrac_mol['Ilm'] = comp_df['TiO2'].tolist()[0]
        comp_df['FeO'] = comp_df['FeO'].tolist()[0] - comp_df['TiO2'].tolist()[0]
        comp_df['TiO2'] = 0.00

    """
    6 Magnetite: Subtract Fe2O3 from FeO. Put the Fe2O3 value in magnetite. Fe2O3 is now zero
    """
    # if 'Fe2O3' in comp_df.columns:
    #     minfrac_mol['Mt'] = comp_df['Fe2O3'].tolist()[0]
    #     comp_df['FeO'] = comp_df['FeO'].tolist()[0] - comp_df['Fe2O3'].tolist()[0]
    #     comp_df['Fe2O3'] = 0.00

    """
    7 Orthoclase: Subtract K2O from Al2O3. Put the K2O value in orthoclase. K2O is now zero
    """
    if comp_df['Al2O3'].tolist()[0] > 0:
        minfrac_mol['Or'] = comp_df['K2O'].values
        comp_df['Al2O3'] = comp_df['Al2O3'] - comp_df['K2O']
        comp_df['K2O'] = 0.00
    else:
        print('Cannot accommodate K because of lack of Al in composition')

    """
    8 Albite (provisional): Subtract Na2O from Al2O3. Put the Na2O value in albite. 
                            Retain the Na2O value for possible normative nepheline.
    """
    if comp_df['Na2O'].values > comp_df['Al2O3'].values:
        minfrac_mol['Ab'] = comp_df['Al2O3'].values
        Na2O_surplus = comp_df['Na2O'] - comp_df['Al2O3']
        comp_df['Al2O3'] = 0
        print(f'There is a Na2O surplus of {Na2O_surplus.values[0]:.2%}!')
    else:
        minfrac_mol['Ab'] = comp_df['Na2O'].values
        comp_df['Al2O3'] = comp_df['Al2O3'] - comp_df['Na2O']

    if verbose_cipw:
        print(f"#8\nAb = {minfrac_mol['Ab'].tolist()[0]}")

    """
    9 Anorthite:                       
    A. If CaO is more than the remaining Al2O3, then subtract Al2O3 from CaO. Put all Al2O3 into Anorthite.

    B. If Al2O3 is more than CaO, then subtract CaO from Al2O3. Put all CaO into anorthite.
    """

    An_CaO = comp_df['CaO'].tolist()[0]  # used for normative An number in step 24

    if verbose_cipw:
        print(f'#9\nCaO = {comp_df["CaO"].tolist()[0]}')
        print(f'#9\nAl2O3 = {comp_df["Al2O3"].tolist()[0]}')
    if comp_df['Al2O3'].tolist()[0] <= comp_df['CaO'].tolist()[0]:
        minfrac_mol['An'] = comp_df['Al2O3'].tolist()[0]
        comp_df['CaO'] = comp_df['CaO'] - comp_df['Al2O3']
        comp_df['Al2O3'] = 0
        if verbose_cipw:
            print(f'#9A\nAn = {minfrac_mol["An"].tolist()[0]}')

    else:
        minfrac_mol['An'] = comp_df['CaO'].tolist()[0]
        comp_df['Al2O3'] = comp_df['Al2O3'] - comp_df['CaO']
        comp_df['CaO'] = 0
        if verbose_cipw:
            print(f'#9B\nAn = {minfrac_mol["An"].tolist()[0]}')

    """
    10 Corundum: If Al2O3 is not zero, put the remaining Al2O3 into corundum.                        
    """
    if comp_df['Al2O3'].tolist()[0] > 0:
        minfrac_mol['Cor'] = comp_df['Al2O3'].tolist()[0]
        comp_df['Al2O3'] = 0.0
        comp_df['CaO'] = comp_df['CaO'] - comp_df['Al2O3']

    """
    11 Calculate Magnesium Number Mg/(Mg+Fe)                       
    """
    if 'FeO' not in comp_df.columns.values.tolist():
        comp_df['FeO'] = 0.00
    magnesium_nbr = comp_df['MgO'].tolist()[0] / (comp_df['MgO'].tolist()[0] + comp_df['FeO'].tolist()[0])

    """
    12. Calculate the mean formula weight of the remaining FeO and MgO. 
        This combined FeMg oxide, called FMO, will be used in subsequent calculations.                     
    """
    FMO_wt = (magnesium_nbr * 40.3044) + ((1 - magnesium_nbr) * 71.8464)

    # Add dictionary entries for final wt%
    amu_min_dict['Opx'] = 60.0843 + FMO_wt
    rho_min_dict['Opx'] = magnesium_nbr * rho_min_dict['En'] + (1 - magnesium_nbr) * rho_min_dict['Fs']
    amu_min_dict['Ol'] = 60.0843 + 2 * FMO_wt
    rho_min_dict['Ol'] = magnesium_nbr * rho_min_dict['Fo'] + (1 - magnesium_nbr) * rho_min_dict['Fa']
    # amu_min_dict['Di'] = 176.2480 + 2 * FMO_wt

    """
    13. Add FeO and MgO to make FMO              
    """

    FMO = comp_df['MgO'].tolist()[0] + comp_df['FeO'].tolist()[0]

    # """
    # 14. Diopside: If CaO is not zero, subtract CaO from FMO. Put all CaO into diopside. CaO is now zero.
    # """
    # if comp_df['CaO'].tolist()[0] > 0:
    #     minfrac_mol['Di'] = comp_df['CaO'].tolist()[0]
    #     FMO = FMO - comp_df['CaO'].tolist()[0]
    #     comp_df['CaO'] = 0.0
    """
    14a. Diopside: If CaO is not zero, set diopside as min(CaO,MgO) and subtract from FMO.                   
    """
    if comp_df['CaO'].tolist()[0] > 0:
        minfrac_mol['Di'] = min(comp_df['CaO'].tolist()[0], comp_df['MgO'].tolist()[0])
        FMO = FMO - minfrac_mol['Di'].tolist()[0]
        comp_df['CaO'] = comp_df['CaO'].tolist()[0] - minfrac_mol['Di'].tolist()[0]

    """
    14b. Diopside: If CaO is not zero, set diopside as min(CaO,MgO) and subtract from MgO.                   
    """
    if comp_df['CaO'].tolist()[0] > 0:
        minfrac_mol['Wo'] = comp_df['CaO'].tolist()[0]
        comp_df['CaO'] = 0.0

    """
    15. Orthopyroxene (provisional): Put all remaining FMO into orthopyroxene. 
        Retain the FMO value for the possible normative olivine.
    """

    if FMO > 0:
        minfrac_mol['Opx'] = FMO
        minfrac_mol['En'] = FMO * magnesium_nbr
        minfrac_mol['Fs'] = FMO * (1 - magnesium_nbr)

    """
    16. Calculate the amount of SiO2 needed for all of the normative silicates listed above, allotting SiO2 as follows:
        Orthoclase * 6 = needed SiO2 for each Orthoclase
        Albite * 6 = needed SiO2 for each Albite
        Anorthite * 2 = needed SiO2 for each Anorthite
        Diopside * 2 = needed SiO2 for each Diopside
        Orthopyroxene * 1 = needed SiO2 for each Hypersthene
    """

    SiO2_Or = minfrac_mol['Or'] * 6
    SiO2_Ab = minfrac_mol['Ab'] * 6
    SiO2_An = minfrac_mol['An'] * 2
    SiO2_Di = minfrac_mol['Di'] * 2
    SiO2_Wo = minfrac_mol['Wo'] * 1
    SiO2_Opx = minfrac_mol['Opx'] * 1

    """
    17. Sum the five SiO2 values just calculated, and call this number pSi1 for the first provisional SiO2.
    """
    pSi1 = SiO2_Or.values + SiO2_Ab.values + SiO2_An.values + SiO2_Di.values + SiO2_Opx.values + SiO2_Wo.values

    """
    18. Quartz: If there is enough silica to make all five minerals in the list in #16 then the
        rock is quartz-normative. Otherwise there is no quartz in the norm and silica to make the rest
        of the silicates must come from other sources.

        A. If pSi1 calculated in #16 is less than SiO2, then there is excess silica. Subtract pSi1 from
        SiO2, and put excess SiO2 in quartz. SiO2, nepheline, and olivine are now zero. Skip to #23.

        B. If pSi1 calculated in #16 is more than SiO2, then the rock is silica deficient. Proceed to #19.
    """

    if comp_df['SiO2'].values < pSi1:
        minfrac_mol['Qz'] = 0.00
        """
        19. -> 20. Sum the four SiO2 values just calculated to get pSi2. Subtract pSi2 from SiO2 to get the
        of SiO2 available for olivine and orthopyroxene, called pSi3.

        A. If FMO is greater than or equal to 2 times pSi3, then put all FMO in Olivine. FMO and
        Orthopyroxene are now zero. Proceed to #21.

        B. If FMO is less than 2 times pSi3, then nepheline is zero. Calculate the amount of
        orhtopyroxene and olivine as follows:
            Orthopyroxne = ((2 * pSi3) - FMO)
            Olivine = (FMO - orthopyroxene)
            Skip to #23
        """
        pSi2 = SiO2_Or.tolist()[0] + SiO2_Ab.tolist()[0] + SiO2_An.tolist()[0] + SiO2_Di.tolist()[0] + \
               SiO2_Wo.tolist()[0]
        pSi3 = comp_df['SiO2'].tolist()[0] - pSi2

        if FMO >= 2 * pSi3:
            if verbose_cipw: print(
                'Total Fe+Mg is greater equal to two times the SiO2 remaining after forming Or, Ab, An and Di ')
            minfrac_mol['Ol'] = FMO
            minfrac_mol['Opx'] = 0.00
            minfrac_mol['En'] = 0.00
            minfrac_mol['Fs'] = 0.00
            FMO = 0

            """
            21. Nepheline, albite (final): If you reached this step, then turning orthopyroxene into olivine
            in #20A did not yield enough silica to make orthoclase, albite, anorthite, diopside, and
            olivine.                 
            """

            SiO2_Ol = 0.5 * minfrac_mol['Ol']

            """
            22. Sum the three SiO2 values just calculated to get pSi4. Subtract pSi4 from SiO2 to get
                pSi5, which is the amount of SiO2 available for albite and nepheline.
                    Albite = (pSi5-(2*Na2O))/4
                    Nepheline = Na2O-Albite
            """
            pSi4 = SiO2_Or.tolist()[0] + SiO2_An.tolist()[0] + SiO2_Di.tolist()[0] + SiO2_Ol.tolist()[0] + \
                   SiO2_Wo.tolist()[0]

            pSi5 = comp_df['SiO2'].tolist()[0] - pSi4
            minfrac_mol['Ab'] = (pSi5 - (2 * comp_df['Na2O'].tolist()[0])) / 4
            if verbose_cipw:
                print(f"#22\nAb = {minfrac_mol['Ab'].tolist()[0]}")
            minfrac_mol['Nph'] = comp_df['Na2O'].tolist()[0] - minfrac_mol['Ab'].tolist()[0]
            if minfrac_mol['Ab'].tolist()[0] < 0:
                raise ValueError('Your composition lies outside the CIPW mineralogy.'
                                 'This may happen if CaO is high and SiO2 low')
            else:
                pass


        elif FMO < 2 * pSi3:
            if verbose_cipw:
                print('Total Fe+Mg is less than two time the SiO2 remaining after forming Or, Ab, An and Di ')
            minfrac_mol['Nph'] = 0.00
            minfrac_mol['Opx'] = (2 * pSi3) - FMO
            minfrac_mol['En'] = ((2 * pSi3) - FMO) * magnesium_nbr
            minfrac_mol['Fs'] = ((2 * pSi3) - FMO) * (1 - magnesium_nbr)
            minfrac_mol['Ol'] = FMO - minfrac_mol['Opx'].tolist()[0]



    elif pSi1 < comp_df['SiO2'].tolist()[0]:
        minfrac_mol['Nph'] = 0
        minfrac_mol['Ol'] = 0
        comp_df['SiO2'] = comp_df['SiO2'] - pSi1
        minfrac_mol['Qz'] = comp_df['SiO2'].tolist()[0]
        comp_df['SiO2'] = 0.0
    """
    23. Multiply orthoclase, albite, and nepheline by two. Divide olivine by two
    """
    minfrac_mol['Or'] = minfrac_mol['Or'].tolist()[0] * 2
    minfrac_mol['Ab'] = minfrac_mol['Ab'].tolist()[0] * 2
    minfrac_mol['Nph'] = minfrac_mol['Nph'].tolist()[0] * 2
    minfrac_mol['Ol'] = minfrac_mol['Ol'].tolist()[0] / 2
    minfrac_mol['Fo'] = minfrac_mol['Ol'].tolist()[0] * magnesium_nbr
    minfrac_mol['Fa'] = minfrac_mol['Ol'].tolist()[0] * (1 - magnesium_nbr)

    # rho_min_dict['Ol'] = rho_min_dict['Fo'] * minfrac_mol['Fo'] + rho_min_dict['Fa'] * minfrac_mol['Fa']

    if verbose_cipw:
        print(f"#23\nAb = {minfrac_mol['Ab'].tolist()[0]}")
    """
    24. Calculate An number, which is the Ca/(Ca+Na) ratio in normative plagioclase:
    """

    An_nbr = comp_df['CaO'].tolist()[0] / (An_CaO + comp_df['Na2O'].tolist()[0])

    """
    25. Plagioclase: Add albite to anorthite to make plagioclase. Retain the albite value, anorthite is now zero.
    """
    minfrac_mol['Plag'] = minfrac_mol['Ab'].tolist()[0] + minfrac_mol['An'].tolist()[0]
    if verbose_cipw:
        print(f"#25\nAb = {minfrac_mol['Ab'].tolist()[0]}")
        print(f"An = {minfrac_mol['An'].tolist()[0]}")
        print(minfrac_mol['Plag'].tolist()[0])

    """
    25. Calculate the formula weight of plagioclase, using the An number value from #24
    """
    plag_wt = (An_nbr * 278.2093) + ((1 - An_nbr) * 262.2230)

    amu_min_dict['Plag'] = plag_wt
    rho_min_dict['Plag'] = An_nbr * rho_min_dict['An'] + (1 - An_nbr) * rho_min_dict['Ab']

    """
    Obtain minfrac for comparison with CIPW (in wt%)
    """
    minfrac_cipw_wt = minfrac_mol.mul(pd.Series(amu_min_dict))

    minfrac_cipw_vol = minfrac_cipw_wt.div(pd.Series(rho_min_dict))
    minfrac_cipw_vol.drop(['Ab', 'An', 'Fo', 'Fa', 'En', 'Fs', 'Wo'], inplace=True, axis=1)
    minfrac_cipw_vol = normalize_dfs([minfrac_cipw_vol])[0]

    """
    Obtain mineralogy in wt% and vol%
    """
    minfrac_mol.drop(['Plag', 'Ol', 'Opx'], inplace=True, axis=1)
    minfrac_wt = minfrac_mol.mul(pd.Series(amu_min_dict))
    minfrac_vol = minfrac_wt.div(pd.Series(rho_min_dict))

    minfrac_mol, minfrac_vol, minfrac_wt = \
        normalize_dfs([minfrac_mol, minfrac_vol, minfrac_wt])

    print(f'\nspubase modal abundances (vol%):\n{minfrac_vol}')
    print(f'Total: {minfrac_vol.sum().iloc[0]:0.2f}')

    if verbose_cipw:
        print(f'\nCIPW modal abundances (vol%):\n{minfrac_cipw_vol}')
        print(f'Total: {minfrac_cipw_vol.sum().iloc[0]:0.2f}')

    self.minfrac_df_weight = minfrac_wt
    self.minfrac_df_volume = minfrac_vol
    self.minfrac_df_volume_CIPW = minfrac_cipw_vol

    return minfrac_vol
