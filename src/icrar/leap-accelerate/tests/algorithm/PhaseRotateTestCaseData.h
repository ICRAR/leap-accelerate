/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#pragma once

#include <casacore/casa/Arrays.h>
#include <casacore/casa/Quanta/MVDirection.h>

#include <vector>
#include <utility>

namespace icrar
{
    /**
     * @brief Get the Expected Calibration object validated the output of LEAP-Cal:ported
     * 
     * @return a vector of direction and antenna calibration pairs
     */
    std::vector<std::pair<casacore::MVDirection, std::vector<double>>> GetExpectedCalibration()
    {
        std::vector<std::pair<casacore::MVDirection, std::vector<double>>> output;
        output.push_back(std::make_pair(casacore::MVDirection(-0.4606549305661674,-0.29719233792392513), std::vector<double>{
             3.47798761718151e-14,
             -0.00648433329323028,
            -2.55740793102962e-14,
            -2.62367361361332e-15,
                0.165497653930812,
               0.0262251479322564,
               0.0443711922081631,
               -0.165578840690953,
              -0.0255160672481987,
            -4.37653088616369e-16,
             1.64979683837035e-14,
             1.05792120184199e-14,
               0.0854092639675605,
               -0.149164639957487,
              -0.0262230128102039,
             5.92557046149321e-15,
                0.182889108680102,
              -0.0438273260052316,
             1.23166967109201e-14,
            -4.41656665786405e-15,
              -0.0350836185696992,
                0.182120930956221,
                0.383094438953473,
               -0.150001187797604,
              -0.0515740095095305,
               -0.163853667233248,
            -9.40867739263726e-15,
               -0.294270876962736,
              -0.0981483406913571,
              -0.0445700989054842,
            -5.92501065971187e-15,
                0.302426723177387,
               -0.211477773208064,
                0.128872731632068,
            -9.79639559886089e-15,
              -0.0613157407205358,
               -0.232780588196737,
               -0.131183447991814,
            -3.80401037503603e-15,
            -1.76488746161417e-15,
               0.0699021707113494,
                0.260414151936181,
                -0.16339492129696,
               0.0757010118310151,
                -0.17381189665743,
               -0.269067063311889,
                 0.19686689377429,
               0.0963317964837667,
               0.0637762512029278,
             5.74949559335027e-15,
                0.242020518374668,
            -1.88501011323325e-15,
               -0.175211019739888,
              -0.0118675322029245,
               -0.116867464457701,
               -0.417429161932406,
               0.0920727339481315,
               -0.205622304194168,
              -0.0560398207793513,
               -0.107066678931721,
             2.05583315931774e-15,
             2.36905785927943e-15,
            -3.02686434723261e-15,
             1.17737689681709e-15,
             9.83282337622787e-16,
              -0.0740559560583796,
                0.183093065942542,
               -0.268899872132513,
              -0.0655457582011908,
                -0.15547454432653,
            -9.78211874838763e-16,
            -1.23158258146447e-15,
               0.0300680891373888,
              -0.0684545893858185,
               -0.241377830490058,
               -0.408466951062746,
              -0.0109464647974038,
              -0.0321404277931541,
                0.314005639514166,
                0.216872408597804,
            -4.22842072184401e-16,
               -0.190490646190885,
               -0.415450572394138,
              -0.0441951756591611,
                0.223551730919814,
               -0.241524852217871,
              -0.0572301756438862,
             4.26634013812075e-15,
                0.165477935146102,
              -0.0992519344330023,
               0.0614250678289765,
            -2.06706513969389e-15,
               -0.266423131966001,
             -2.6118680925951e-15,
              1.8983168779605e-15,
             1.12131982546335e-15,
                0.096973955874553,
                0.130163787303748,
                0.159103162429766,
               -0.238441793255204,
               -0.177205350383795,
                0.159037527291524,
              -0.0413278074008709,
               -0.297045429616852,
                 -0.1300341709104,
                0.172030776410581,
               0.0940368261934581,
               0.0733889029034323,
                0.328937108162603,
               0.0906952590932191,
               -0.389925720652089,
             1.65074831516921e-16,
                0.127676411232404,
                0.253362853246073,
               -0.245322409873246,
            -3.77930088037526e-16,
               0.0875545333444695,
                -0.16120440688403,
               -0.259456587539171,
               -0.232294482843081,
              0.00573935369679734,
               -0.289167715997086,
               0.0236250500579862,
                0.110080500451047,
                -0.35207552782245,
            -0.000161579240933296,
               -0.024539812322464,
               -0.148097053475697,
        }));

        // output.push_back(std::make_pair(casacore::MVDirection(-0.753231018062671,-0.44387635324622354), std::vector<double>{
        //     2.66940707100503,
        //      2.2970261355245,
        //                    0,
        //                    0,
        //     2.37286363595303,
        //     2.63169072474505,
        //     2.72339153492401,
        //      2.4787964771248,
        //     2.56266530776803,
        //                    0,
        //                    0,
        //                    0,
        //      2.3498107749363,
        //     2.60812066866685,
        //     2.71155466852463,
        //                    0,
        //     2.53239692415824,
        //     2.59639746435725,
        //                    0,
        //                    0,
        //     2.68815507344987,
        //     2.42280642919287,
        //     2.38381829083625,
        //     2.17897172810935,
        //     2.59142871786394,
        //     2.45175822369597,
        //                    0,
        //     2.33658232492272,
        //     2.62501940297269,
        //     2.50845523780901,
        //                    0,
        //     2.45800221101849,
        //     2.11690589644253,
        //     2.54346336226205,
        //                    0,
        //     2.77596376574356,
        //     2.59632501770404,
        //     2.58941712994333,
        //                    0,
        //                    0,
        //     2.47954340799536,
        //     2.78908831120321,
        //     2.58855227550415,
        //     2.34748400404489,
        //     2.84913863482906,
        //     2.12753070636066,
        //     2.05852397500549,
        //     2.26811193090935,
        //     2.67369416525684,
        //                    0,
        //      2.7507484091406,
        //                    0,
        //     2.58484600698607,
        //     2.26696309266383,
        //     2.39715817905781,
        //     2.74973915060334,
        //     2.51109323679793,
        //     2.58314298840669,
        //     2.45463216062136,
        //     2.29311157172345,
        //                    0,
        //                    0,
        //                    0,
        //                    0,
        //                    0,
        //     2.73450051928645,
        //     2.33848299978127,
        //     2.49046430796644,
        //     2.49370429585425,
        //     2.45176370212733,
        //                    0,
        //                    0,
        //     2.33338397930632,
        //     2.64931914296968,
        //     2.41145476301061,
        //     2.55940951035906,
        //     2.44728828799146,
        //     2.65642398536674,
        //      2.5245266383623,
        //     2.40560958368183,
        //                    0,
        //     2.11783575121744,
        //     2.49265370256483,
        //     2.33702500242863,
        //     2.22231181478633,
        //     2.59083932865882,
        //     2.55754303688411,
        //                    0,
        //     2.43892301312037,
        //     2.60771128744822,
        //     2.20792262714726,
        //                    0,
        //     2.56329491748639,
        //                    0,
        //                    0,
        //                    0,
        //       2.622212205822,
        //     2.56653379895624,
        //     2.66296476018488,
        //     2.56776882414028,
        //     2.56062742370982,
        //     2.82979723755799,
        //     2.43247977790005,
        //     2.52792625983831,
        //     2.31189781420646,
        //     2.68471348792089,
        //     2.52144021308095,
        //     2.81641633041315,
        //     2.66123518865772,
        //     2.37457923334427,
        //     2.83433133594982,
        //                    0,
        //      2.6096572778671,
        //     2.54270676897643,
        //     2.86993859813482,
        //                    0,
        //     2.70824021037875,
        //     2.43279237964071,
        //     2.39891158028436,
        //     2.38212304452993,
        //     2.39291435669244,
        //     2.25098998284473,
        //     2.63989589493289,
        //       2.431640263158,
        //     2.62321928177316,
        //     2.50927244546791,
        //     2.41362292454067,
        //     2.76440352155434,
        // }));

        // output.push_back(std::make_pair(casacore::MVDirection(-0.6207547100721282,-0.2539086572881469), std::vector<double>{
        //     3.02649528654214,
        //     3.10777887684405,
        //                    0,
        //                    0,
        //     2.81212632484272,
        //     2.88622529752106,
        //     3.20642545507654,
        //     2.82456265927589,
        //     3.02575915917859,
        //                    0,
        //                    0,
        //                    0,
        //      3.1468930917008,
        //     3.41648740549913,
        //     2.81273653786727,
        //                    0,
        //     2.95389528330968,
        //     2.95948996876507,
        //                    0,
        //                    0,
        //     2.90832680822659,
        //     3.15549292896688,
        //     2.96158096881739,
        //     3.14581570375654,
        //     2.94261706130072,
        //      3.3571399345161,
        //                    0,
        //     2.99301301501561,
        //     3.32630143618634,
        //     3.13819551153512,
        //                    0,
        //     3.05566904198689,
        //     3.14975711830892,
        //     2.89412773329797,
        //                    0,
        //     2.91326740419561,
        //     2.78177196652179,
        //     3.10098779461005,
        //                    0,
        //                    0,
        //     3.02818851094753,
        //     3.07908014019085,
        //     2.92759146580687,
        //     2.98047661247693,
        //     3.17173850491018,
        //     2.97402493811016,
        //     2.65521788178272,
        //     3.00233761241966,
        //     2.63263034009139,
        //                    0,
        //     2.91383760331526,
        //                    0,
        //     2.93930253762704,
        //     2.86873308968946,
        //     2.96328606254811,
        //     3.14720054483507,
        //     3.15999757850614,
        //     3.01573011395736,
        //     2.91230524663411,
        //     2.82302605259917,
        //                    0,
        //                    0,
        //                    0,
        //                    0,
        //                    0,
        //     3.11736370155431,
        //     3.02641965575357,
        //      3.1974744023122,
        //     3.07507911838769,
        //     2.72356631729012,
        //                    0,
        //                    0,
        //     3.06277693715546,
        //     2.89659778792819,
        //      3.1604058232977,
        //     2.86765938016416,
        //      2.9715891719881,
        //     3.18701757291299,
        //     3.03832735949693,
        //     2.93024942866138,
        //                    0,
        //     2.78778505030499,
        //     3.03617080794071,
        //     3.29384451343836,
        //     2.91872345871414,
        //     2.74264674796115,
        //     3.15189951906102,
        //                    0,
        //     3.10524831580181,
        //     2.93001675089563,
        //     2.95582807470878,
        //                    0,
        //     3.00883801697744,
        //                    0,
        //                    0,
        //                    0,
        //     2.80220655192013,
        //     2.85734363385934,
        //      3.1022812874959,
        //     3.01633464585403,
        //     3.06104307479965,
        //     2.84158971978347,
        //     2.67950574434861,
        //     3.08206033468821,
        //     2.69357120801818,
        //     2.66772364538293,
        //     2.96552386218769,
        //     3.04092420222566,
        //     2.95456394545774,
        //      3.1103778624671,
        //     3.26807067488729,
        //                    0,
        //     3.24518992619425,
        //      2.7046261731372,
        //     3.12206182399197,
        //                    0,
        //     2.99429114041317,
        //     2.77900859216216,
        //     2.94083236294281,
        //     3.04405219905266,
        //     2.91723739526017,
        //      2.6466651567884,
        //     2.88554172053832,
        //     3.24423565983191,
        //     2.93071313385988,
        //     2.92675261197971,
        //     3.02495480201382,
        //     3.08042209050104,
        // }));

        // output.push_back(std::make_pair(casacore::MVDirection(-0.41958660604621867,-0.03677626900108552), std::vector<double>{
        //     2.09405670300514,
        //     2.24973265632178,
        //                    0,
        //                    0,
        //     2.19814491003888,
        //     1.98577745175181,
        //     1.79954717619448,
        //       2.375570947815,
        //     1.96192668848045,
        //                    0,
        //                    0,
        //                    0,
        //     2.30337558595535,
        //     2.07818019810956,
        //      2.2444819218512,
        //                    0,
        //     2.32663604895753,
        //     2.17930895554649,
        //                    0,
        //                    0,
        //     2.26311144882305,
        //     2.38321075715048,
        //     2.37140844870669,
        //      2.0979621334619,
        //     2.23308944656673,
        //     1.67361839723636,
        //                    0,
        //      2.4803843973393,
        //     1.93453981153673,
        //     2.16705104199178,
        //                    0,
        //     2.20742262459279,
        //     2.37353217394698,
        //     2.37859400915076,
        //                    0,
        //      2.4394637411842,
        //      2.0328638105279,
        //     2.28405537297735,
        //                    0,
        //                    0,
        //     1.94772924219363,
        //     2.43682195394374,
        //     2.02707166314488,
        //      2.1497004813737,
        //     2.22460505098086,
        //     2.04703407589749,
        //      2.0334889934639,
        //     2.18527367359791,
        //     1.98228479907915,
        //                    0,
        //      2.3315541351602,
        //                    0,
        //     2.28303319068191,
        //     1.96740369544605,
        //     2.36242832181987,
        //     2.06570314724028,
        //     2.02575000573757,
        //     2.20320929024714,
        //     2.28709092639526,
        //     2.40266193725367,
        //                    0,
        //                    0,
        //                    0,
        //                    0,
        //                    0,
        //      2.1493691505881,
        //     1.92442968518846,
        //     1.77336393210406,
        //     2.16373879913863,
        //     2.33362254461566,
        //                    0,
        //                    0,
        //      2.0579547235385,
        //     2.15238576718379,
        //     2.11355895034468,
        //     2.03559362218414,
        //     2.39790919746143,
        //      2.0580525191644,
        //     2.05420556600375,
        //     2.02407580569069,
        //                    0,
        //     2.17577991673266,
        //      1.9937266688611,
        //      1.8768316594321,
        //     2.19972627612467,
        //     2.36440819952834,
        //     2.02857839961879,
        //                    0,
        //     2.25667910985571,
        //     1.88301437871967,
        //     1.99932852462542,
        //                    0,
        //     1.92429679062874,
        //                    0,
        //                    0,
        //                    0,
        //     1.96492474429743,
        //     2.09041892569708,
        //     1.83411862358093,
        //      2.3316965560831,
        //     1.99441690246242,
        //     2.04991828686767,
        //      2.0706823318835,
        //     2.06337436773907,
        //     2.28941978927821,
        //     2.22215350498702,
        //     2.32674373682571,
        //     1.88901431693404,
        //     2.41964518104566,
        //     1.93843961245602,
        //     2.39197820612532,
        //                    0,
        //     2.19622982806276,
        //      2.1235354712062,
        //     2.29497953216884,
        //                    0,
        //     2.17389078599935,
        //     2.38290169020514,
        //     2.37641930189825,
        //     2.28004005265543,
        //     2.09104778481841,
        //       2.262270476462,
        //     1.75961070307986,
        //     2.11913834969633,
        //      2.1298777398892,
        //     2.16337478303729,
        //     2.44521328711319,
        //     2.25694596891932,
        // }));
        return output;
    }
}