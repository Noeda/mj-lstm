{-
 - This Haskell file replicates the implementation of LSTMv2 in src/lstm_v2.rs
 -
 - I've used it to verify the gradient computation in the Rust code is correct.
 - Haskell has easy-to-use but slow automatic reverse mode differentiation,
 - particularly useful to check that other code is correct.
 -}

{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveAnyClass #-}

module Main
  ( main )
  where

import Data.Aeson
import Data.Text ( Text )
import Data.Vector ( Vector, (!) )
import qualified Data.Vector as V
import Numeric.AD

fastSigmoid :: Floating a => a -> a
fastSigmoid x = 0.5 + (x / (1 + abs x)) / 2

data IIOF a = IIOF !a !a !a !a
  deriving ( Eq, Ord, Show, Functor, Traversable, Foldable )

instance Num a => Num (IIOF a) where
  IIOF a b c d + IIOF e f g h = IIOF (a+e) (b+f) (c+g) (d+h)
  IIOF a b c d - IIOF e f g h = IIOF (a-e) (b-f) (c-g) (d-h)
  IIOF a b c d * IIOF e f g h = IIOF (a*e) (b*f) (c*g) (d*h)
  abs (IIOF a b c d) = IIOF (abs a) (abs b) (abs c) (abs d)
  signum (IIOF a b c d) = IIOF (signum a) (signum b) (signum c) (signum d)
  fromInteger i = IIOF (fromInteger i) (fromInteger i) (fromInteger i) (fromInteger i)

iiof :: a -> IIOF a
iiof x = IIOF x x x x

data LSTM a = LSTM {
    toWeights :: Vector (Vector (IIOF a))
  , selfWeights :: Vector (Vector (IIOF a))
  , outputWeights :: Vector a
  , outputBiases :: Vector a
  , biases :: Vector (Vector (IIOF a))
  }
  deriving ( Eq, Ord, Show, Functor, Traversable, Foldable )

iioify :: [a] -> Vector (IIOF a)
iioify (a:b:c:d:rest) = V.cons (IIOF a b c d) (iioify rest)
iioify [] = V.empty

l :: [Double]
l = [-0.2327889162928023,-0.6394861042581406,-0.48954538467951725,-0.8202478286898756,-0.2317835151804437,0.6581415077640655,-0.4075712490835879,0.340285857864858,0.7809254078359689,-0.44043920614361065,0.7752130995722899,-0.8600977384320898,-0.5152724143165468,0.7489705114072285,0.12225833480707182,0.9358192149530682,-0.17227620921885434,0.41767689997401636,-0.770700113624434,0.9503792923977596,0.9018072146264529,-0.16288352505358716,-0.20936005018812232,0.2289363198398302,-0.9410276941841165,0.9451692579488808,0.4942033520900275,-0.6287265337099521,0.09585445411724836,0.34485325203426864,0.09857404160221517,0.8167814850330397,0.7732228124353182,0.7513473875121326,0.3024431220783721,-0.19422812256282684,0.4513089879463914,-0.21431724304522337,-0.9509663938150865,-0.970385542532016,-0.3026246998536619,0.6139209736323483,0.0809311383196949,-0.8418028761291168,-0.8012889582521634,-0.3935103500038766,0.2202342319786621,-0.006250895764976505,-0.7219955917798742,-0.452373052263801,0.5038228920447385,0.3253688157685697,-0.889943673752478,-0.733319092050619,0.18969367443636154,-0.6749614294023041,-0.2856994498583454,-0.7210635060605952,-0.7522465220516286,-0.11060799576907021,-0.8843212547232162,0.4814133539583083,0.8351927367609333,-0.3692216358619662,0.7623883852790714,0.6307563007872932,-0.2727832245472106,-0.5211725850328475,0.7788158995754615,-0.8765121856514035,-0.2883888444343947,0.11107227319381918,0.7007954527717386,-0.2517658189361849,-0.13930259257986854,-0.9992880650974767,0.14415288615722233,-0.42187382602638124,-0.8211480153533155,-0.6987810951636328,0.1715165172876949,0.7944990067161966,0.7168488110608009,0.9304546548828521,-0.4864964449538909,0.736716372671478,-0.48367275412873134,-0.7092577235494071,-0.2904707991155431,-0.8468612445735988,-0.3491550960772196,-0.7837747936999602,0.1368396062497621,0.7469959676920457,0.5821333683229826,0.023047176018382576,0.5315559091227127,-0.41219643655521887,0.9257360368688388,-0.06038584539280212,-0.07084210346619813,0.7622224521890262,-0.49778035674739973,0.44660487590123443,-0.21777999885522048,0.584524561584447,-0.8196473306647918,-0.7870096005236205,-0.8375113448972571,0.7610850967249676,0.6178477461728606,0.7951755094046269,-0.6336996401083788,-0.9222391886518229,0.2519421469795948,0.3937590164412894,-0.6621622009416499,0.4285940808773674,0.8968104455197556,-0.7208810165627595,0.796023143471583,0.939410772198177,-0.3673281371322936,0.6025509768756341,0.5435318921350292,0.5151812050368769,0.9814764871701493,0.6217368057939101]

leftover4 :: [a] -> Int
leftover4 lst =
  4 - (length lst `mod` 4)

fromVec :: [Double] -> [Int] -> LSTM Double
fromVec vals' layer_sizes =
  let vals = vals'
      (tsbm, rest) = go vals layer_sizes
      (out_weights, out_biases) = go2 rest (last layer_sizes) (last $ init layer_sizes)

      to_weights = fmap (\(a, _, _) -> iioify a) tsbm
      self_weights = fmap (\(_, a, _) -> iioify a) tsbm
      biases = fmap (\(_, _, a) -> iioify a) tsbm

   in LSTM {
       toWeights = V.fromList to_weights,
       selfWeights = V.fromList self_weights,
       outputWeights = V.fromList out_weights,
       outputBiases = V.fromList out_biases,
       biases = V.fromList biases }
 where
  go :: [Double] -> [Int] -> ([([Double], [Double], [Double])], [Double])
  go vals (src_sz:tgt_sz:tgt2_sz:rest) =
    -- First it's weights from previous layer to this layer,
    -- then it's weights from layer to itself (self weights),
    -- and then it's bias parameters
    -- and then it's memory cell parameters
    let (to_wgts, rest1) = splitAt (src_sz * tgt_sz * 4) vals
        (self_wgts, rest2) = splitAt (tgt_sz * tgt_sz * 4) rest1
        (biases, rest3) = splitAt (tgt_sz * 4) rest2
        (rest_params, rest4) = go rest3 (tgt_sz:tgt2_sz:rest)
     in ((to_wgts, self_wgts, biases):rest_params, rest4)
  go rest [_src_sz, _tgt_sz] = ([], rest)

  go2 vals tgt_sz src_sz =
    let (wgts, rest1) = splitAt (tgt_sz * src_sz) vals
        (biases, rest2) = splitAt tgt_sz rest1
     in (wgts, biases)

data LSTMState a = LSTMState {
    nn :: LSTM a
  , lastActivations :: Vector (Vector a)
  , memories :: Vector (Vector a)
  }
  deriving ( Eq, Ord, Show )

lstm :: Num a => [Int] -> LSTM a
lstm layer_sizes = LSTM
  {
    toWeights = generateSz layer_sizes $ \src tgt -> V.replicate (src * tgt) (iiof 0),
    selfWeights = generateSz layer_sizes $ \_ tgt -> V.replicate (tgt * tgt) (iiof 0),
    outputWeights = V.replicate (last layer_sizes * last (init layer_sizes)) 0,
    outputBiases = V.replicate (last layer_sizes) 0,
    biases = generateSz layer_sizes $ \_ tgt -> V.replicate tgt (iiof 0)
  }

start :: Num a => LSTM a -> LSTMState a
start lstm = LSTMState
  {
    nn = lstm,
    lastActivations = V.generate (length $ biases lstm) $ \i -> V.replicate (V.length $ biases lstm V.! i) 0,
    memories = V.generate (length $ biases lstm) $ \i -> V.replicate (V.length $ biases lstm V.! i) 0
  }

foldI :: Num a => Int -> (Int -> a) -> a
foldI num_items action = go 0
 where
  go i | i < num_items = action i + go (i+1)
       | otherwise = 0

propagate :: forall a. (Floating a, Num a) => LSTMState a -> Vector a -> (LSTMState a, Vector a)
propagate state inputs =
  let new_act_memories = go inputs 0
      final_activations = go2 (fst $ last new_act_memories)

   in (state
        { lastActivations = V.fromList (fmap fst new_act_memories)
        , memories = V.fromList (fmap snd new_act_memories) }
      , final_activations)
 where
  go2 prev_layer_activations =
    let num_targets = V.length (outputBiases (nn state))
        num_sources = V.length prev_layer_activations

        wgt = outputWeights (nn state)
        b = outputBiases (nn state)

     in V.generate num_targets $ \tgt_idx ->
          let act1 = b ! tgt_idx + foldI num_sources (\src_idx ->
                         prev_layer_activations ! src_idx * (wgt ! (src_idx + tgt_idx * num_sources)))

           in fastSigmoid act1

  go :: Vector a -> Int -> [(Vector a, Vector a)]
  go prev_layer_activations layer_idx | layer_idx < V.length (biases (nn state)) =
    let num_targets = V.length (biases (nn state) ! layer_idx)
        num_sources = V.length prev_layer_activations

        wgt = toWeights (nn state) ! layer_idx
        self_wgt = selfWeights (nn state) ! layer_idx
        last_activations = lastActivations state ! layer_idx
        old_memories = memories state ! layer_idx

        b = biases (nn state) ! layer_idx

        (new_acts, new_memories) = generate2 num_targets $ \tgt_idx ->
          let act1 :: IIOF a
              act1 = b ! tgt_idx + foldI num_sources (\src_idx ->
                           iiof (prev_layer_activations ! src_idx) * (wgt ! (src_idx + tgt_idx * num_sources)))
                         + foldI num_targets (\src_idx ->
                             iiof (last_activations ! src_idx) * (self_wgt ! (src_idx + tgt_idx * num_targets)))

              IIOF inp' inp_gate' out' for' = act1
              inp = fastSigmoid inp' * 2 - 1
              inp_gate = fastSigmoid inp_gate'
              out = fastSigmoid out'
              for = fastSigmoid for'

              old_memory = old_memories ! tgt_idx
              new_memory = inp * inp_gate + old_memory * for
              new_activation = (fastSigmoid new_memory * 2 - 1) * out

           in (new_activation, new_memory)

      in (new_acts, new_memories):go new_acts (layer_idx+1)

  go last_activations _ = []

generate2 :: Int -> (Int -> (a, b)) -> (Vector a, Vector b)
generate2 n generator =
  let vec = V.generate n generator
   in (V.map fst vec, V.map snd vec)

generateSz :: [Int] -> (Int -> Int -> a) -> Vector a
generateSz layer_sizes generator = V.fromList $ go layer_sizes
 where
  go [] = []
  go [_] = []
  go (x:y:rest)
    = generator x y:go (y:rest)

main :: IO ()
main = do
  print nn
  --print $ snd $ propagate st (V.fromList [0.311, 0.422])

  let g = grad (\nn -> let st = start nn
                           (st2, v) = propagate st (V.fromList [0.311, 0.422])
                        in V.sum (snd $ propagate st2 (V.fromList [0.422, 0.311])) +
                           V.sum v) nn
  print g
 where
  nn = fromVec l [2, 2, 2, 2, 2]

  st = start nn
