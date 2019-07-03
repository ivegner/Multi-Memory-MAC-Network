# Notes on performance

## Vanilla MAC
Epoch: 1; Loss: 0.79425; Acc: 0.59646: [30:38<00:00,  5.90it/s]
Avg Acc: 0.60788
Epoch: 2; Loss: 0.56873; Acc: 0.72860: [31:27<00:00,  5.79it/s]
Avg Acc: 0.76047
Epoch: 3; Loss: 0.30509; Acc: 0.86179: [31:08<00:00,  6.21it/s]
Avg Acc: 0.90701
Epoch: 4; Loss: 0.23143; Acc: 0.88911: [31:15<00:00,  6.57it/s]
Avg Acc: 0.92354
Epoch: 5; Loss: 0.28529; Acc: 0.90738: [31:37<00:00,  5.76it/s]
Avg Acc: 0.93522
Epoch: 6; Loss: 0.10179; Acc: 0.91826: [31:18<00:00,  5.82it/s]
Avg Acc: 0.94170
...
At 10 epochs, accuracy is supposed to be 95.75%
At 20, 96.5%

# 3 Memories (n=3) [times are on one GPU]
Epoch: 1; Loss: 0.91446; Acc: 0.51202: [1:07:42<00:00,  2.93it/s]
Avg Acc: 0.53016
Epoch: 2; Loss: 0.71245; Acc: 0.61467: [1:07:51<00:00,  3.05it/s]
Avg Acc: 0.63245
Epoch: 3; Loss: 0.41644; Acc: 0.84885: [1:08:15<00:00,  3.25it/s]
Avg Acc: 0.88836
Epoch: 4; Loss: 0.32514; Acc: 0.89475: [1:07:42<00:00,  3.35it/s]
Avg Acc: 0.92665
Epoch: 5; Loss: 0.11384; Acc: 0.91390: [1:07:45<00:00,  2.43it/s]
Avg Acc: 0.94342
Epoch: 6; Loss: 0.10911; Acc: 0.92774: [1:07:27<00:00,  3.04it/s]
Avg Acc: 0.94999
Epoch: 7; Loss: 0.03950; Acc: 0.93569: [1:07:30<00:00,  2.93it/s]
Avg Acc: 0.95545
Epoch: 8; Loss: 0.14394; Acc: 0.94439: [1:07:12<00:00,  3.39it/s]
Avg Acc: 0.95871
Epoch: 9; Loss: 0.06399; Acc: 0.94976: [1:08:19<00:00,  3.10it/s]
Avg Acc: 0.96146
Epoch: 10; Loss: 0.07438; Acc: 0.95375: [1:07:37<00:00,  2.86it/s]
Avg Acc: 0.96279
Epoch: 11; Loss: 0.24445; Acc: 0.95766: [1:08:20<00:00,  3.18it/s]
Avg Acc: 0.96477
Epoch: 12; Loss: 0.02926; Acc: 0.95835: [1:07:57<00:00,  3.02it/s]
Avg Acc: 0.96592
Epoch: 13; Loss: 0.05796; Acc: 0.96424: [1:08:02<00:00,  3.04it/s]
Avg Acc: 0.96622
Epoch: 14; Loss: 0.01756; Acc: 0.96504: [1:07:58<00:00,  3.00it/s]
Avg Acc: 0.96709
Epoch: 15; Loss: 0.00573; Acc: 0.96683: [1:08:11<00:00,  2.95it/s]
Avg Acc: 0.96742
Epoch: 16; Loss: 0.02573; Acc: 0.96864: [1:08:08<00:00,  3.33it/s]
Avg Acc: 0.96834

## 7 Memories (n=7)
Epoch: 1; Loss: 0.65650; Acc: 0.60635: [1:30:01<00:00,  2.13it/s]
Avg Acc: 0.61422
Epoch: 2; Loss: 0.41662; Acc: 0.72297: [1:30:25<00:00,  2.25it/s]
Avg Acc: 0.75226
Epoch: 3; Loss: 0.30118; Acc: 0.86245: [1:30:34<00:00,  2.32it/s]
Avg Acc: 0.90065
Epoch: 4; Loss: 0.19540; Acc: 0.89655: [1:30:26<00:00,  2.18it/s]
Avg Acc: 0.92178
Epoch: 5; Loss: 0.19991; Acc: 0.91657: [1:30:33<00:00,  2.09it/s]
Avg Acc: 0.94092
Epoch: 6; Loss: 0.17701; Acc: 0.93012: [1:30:38<00:00,  2.30it/s]
Avg Acc: 0.94876
Epoch: 7; Loss: 0.06979; Acc: 0.94454: [1:30:33<00:00,  2.14it/s]
Avg Acc: 0.95222
Epoch: 8; Loss: 0.07817; Acc: 0.94772: [1:30:25<00:00,  2.22it/s]
Avg Acc: 0.95426
Epoch: 9; Loss: 0.02039; Acc: 0.95195: [1:30:28<00:00,  2.19it/s]
Avg Acc: 0.95660
Epoch: 10; Loss: 0.12727; Acc: 0.95570: [1:31:24<00:00,  2.21it/s]
Avg Acc: 0.95769
Epoch: 11; Loss: 0.01333; Acc: 0.95767: [1:31:17<00:00,  2.20it/s]
Avg Acc: 0.95894
Epoch: 12; Loss: 0.00793; Acc: 0.96058: [1:31:17<00:00,  2.29it/s]
Avg Acc: 0.95992
Epoch: 13; Loss: 0.02927; Acc: 0.96416: [1:31:18<00:00,  2.28it/s]
Avg Acc: 0.96022
Epoch: 14; Loss: 0.01389; Acc: 0.96606: [1:31:16<00:00,  2.16it/s]
Avg Acc: 0.96066
Epoch: 15; Loss: 0.04256; Acc: 0.96597: [1:31:15<00:00,  2.36it/s]
Avg Acc: 0.96186
Epoch: 16; Loss: 0.19341; Acc: 0.96905: [1:31:19<00:00,  1.72it/s]
Avg Acc: 0.96156
Epoch: 17; Loss: 0.24344; Acc: 0.97148: [1:31:16<00:00,  2.21it/s]
Avg Acc: 0.96213
Epoch: 18; Loss: 0.00172; Acc: 0.96842: [1:31:14<00:00,  2.06it/s]
Avg Acc: 0.96177
Epoch: 19; Loss: 0.01097; Acc: 0.97169: [1:31:13<00:00,  2.01it/s]
Avg Acc: 0.96233
Epoch: 20; Loss: 0.00028; Acc: 0.97252: [1:31:16<00:00,  2.29it/s]
Avg Acc: 0.96270

# 10-cell MAC
Epoch: 2; Loss: 0.88696; Acc: 0.48939: [23:19<00:00,  3.82it/s]
Avg Acc: 0.47681
Epoch: 3; Loss: 0.87297; Acc: 0.50860: [23:29<00:00,  4.16it/s]
Avg Acc: 0.49574
Epoch: 4; Loss: 0.74861; Acc: 0.54529: [23:24<00:00,  4.19it/s]
Avg Acc: 0.52673
Epoch: 5; Loss: 0.61717; Acc: 0.61448: [23:16<00:00,  3.68it/s]
Avg Acc: 0.59394
Epoch: 6; Loss: 0.60718; Acc: 0.63744: [23:27<00:00,  4.11it/s]
Avg Acc: 0.62370
Epoch: 7; Loss: 0.56248; Acc: 0.66192: [23:20<00:00,  4.22it/s]
Avg Acc: 0.63901
Epoch: 8; Loss: 0.47629; Acc: 0.68284: [23:14<00:00,  3.75it/s]
Avg Acc: 0.64655
Epoch: 9; Loss: 0.50212; Acc: 0.69319: [23:21<00:00,  4.34it/s]
Avg Acc: 0.65693
Epoch: 10; Loss: 0.50404; Acc: 0.70245: [23:26<00:00,  3.81it/s]
Avg Acc: 0.65513
Epoch: 11; Loss: 0.43532; Acc: 0.71525: [23:27<00:00,  3.79it/s]
Avg Acc: 0.66435
Epoch: 12; Loss: 0.44736; Acc: 0.72568: [23:28<00:00,  4.40it/s]
Avg Acc: 0.66767
Epoch: 13; Loss: 0.47610; Acc: 0.73740: [23:18<00:00,  4.16it/s]
Avg Acc: 0.66925
Epoch: 14; Loss: 0.44108; Acc: 0.75496: [23:17<00:00,  3.81it/s]
Avg Acc: 0.68775
Epoch: 15; Loss: 0.38218; Acc: 0.79661: [23:17<00:00,  4.27it/s]
Avg Acc: 0.72594
Epoch: 16; Loss: 0.30228; Acc: 0.84962: [23:27<00:00,  3.12it/s]
Avg Acc: 0.79575
Epoch: 17; Loss: 0.30702; Acc: 0.86732: [23:23<00:00,  3.83it/s]
Avg Acc: 0.80723
Epoch: 18; Loss: 0.29626; Acc: 0.87913: [23:21<00:00,  3.85it/s]
Avg Acc: 0.80511
Epoch: 19; Loss: 0.21234; Acc: 0.89393: [23:19<00:00,  3.33it/s]
Avg Acc: 0.83490
Epoch: 20; Loss: 0.22778; Acc: 0.90398: [23:28<00:00,  4.38it/s]
Avg Acc: 0.84120
Epoch: 21; Loss: 0.19717; Acc: 0.91572: [23:31<00:00,  3.78it/s]
Avg Acc: 0.84946
Epoch: 22; Loss: 0.13759; Acc: 0.91691: [23:29<00:00,  3.85it/s]
Avg Acc: 0.85068
...
Epoch: 55; Loss: 0.00749; Acc: 0.98522:[23:39<00:00,  4.17it/s]
Avg Acc: 0.87075
...
Epoch: 75; Loss: 0.07726; Acc: 0.99114: [23:32<00:00,  3.89it/s]
Avg Acc: 0.86807

# 32-cell MAC (classifier enabled during training)
Epoch: 1; Loss: 0.93933; Acc: 0.43913 [27:34<00:00,  3.55it/s]
Avg Acc: 0.40672
...
Epoch: 3; Loss: 0.85783; Acc: 0.49005 [27:46<00:00,  3.36it/s]
Avg Acc: 0.41048
Epoch: 4; Loss: 0.84266; Acc: 0.50313 [27:30<00:00,  3.25it/s]
Avg Acc: 0.39640
Epoch: 5; Loss: 0.81515; Acc: 0.51352 [27:35<00:00,  3.33it/s]
Avg Acc: 0.40534
Epoch: 6; Loss: 0.80390; Acc: 0.52346 [27:41<00:00,  3.48it/s]
Avg Acc: 0.45227
...
Epoch: 16; Loss: 0.68252; Acc: 0.58936 [27:33<00:00,  3.22it/s]
Avg Acc: 0.48780
Epoch: 17; Loss: 0.71680; Acc: 0.59845 [27:28<00:00,  3.71it/s]
Avg Acc: 0.48814
Epoch: 18; Loss: 0.71462; Acc: 0.60698 [27:50<00:00,  3.54it/s]
Avg Acc: 0.47626
Epoch: 19; Loss: 0.72581; Acc: 0.62089 [27:49<00:00,  3.63it/s]
Avg Acc: 0.48062
Epoch: 20; Loss: 0.65033; Acc: 0.62814 [27:38<00:00,  2.78it/s]
Avg Acc: 0.48284

# 10-cell MAC with matrix transformations in cells
Epoch: 1; Loss: 0.89526; Acc: 0.49100: [1:24:48<00:00,  1.11it/s]
Avg Acc: 0.48880
Epoch: 2; Loss: 0.81675; Acc: 0.51316: [1:23:15<00:00,  1.15it/s]
Avg Acc: 0.50738
Epoch: 3; Loss: 0.76692; Acc: 0.53799: [1:23:09<00:00,  1.13it/s]
Avg Acc: 0.53126
Epoch: 4; Loss: 0.65962; Acc: 0.61479: [1:23:15<00:00,  1.15it/s]
Avg Acc: 0.58704
Epoch: 5; Loss: 0.55009; Acc: 0.66590: [1:23:11<00:00,  1.15it/s]
Avg Acc: 0.64791
Epoch: 6; Loss: 0.47732; Acc: 0.69377: [1:23:13<00:00,  1.09it/s]
Avg Acc: 0.68261
Epoch: 7; Loss: 0.49707; Acc: 0.71458: [1:23:05<00:00,  1.16it/s]
Avg Acc: 0.70168
Epoch: 8; Loss: 0.45677; Acc: 0.73128: [1:23:18<00:00,  1.12it/s]
Avg Acc: 0.71738
Epoch: 9; Loss: 0.50999; Acc: 0.74680: [1:23:13<00:00,  1.16it/s]
Avg Acc: 0.73098
Epoch: 10; Loss: 0.50410; Acc: 0.77228: [1:23:14<00:00,  1.15it/s]
Avg Acc: 0.74987
Epoch: 11; Loss: 0.44575; Acc: 0.79254: [1:23:10<00:00,  1.15it/s]
Avg Acc: 0.76049