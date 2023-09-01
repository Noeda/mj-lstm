{-
 - This Haskell file replicates the implementation of LSTMv2 in src/lstm_v2.rs
 -
 - I've used it to verify the gradient computation in the Rust code is correct.
 - Haskell has easy-to-use but slow automatic reverse mode differentiation,
 - particularly useful to check that other code is correct.
 -}

{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveAnyClass #-}

module Main
  ( main )
  where

import Data.Aeson
import qualified Data.ByteString.Lazy as BL
import Data.Text ( Text )
import Data.Vector ( Vector, (!) )
import qualified Data.Vector as V
import GHC.Generics
import Numeric.AD
import System.IO.Unsafe
import Debug.Trace

fastSigmoid :: Floating a => a -> a
fastSigmoid x = 0.5 + (x / (1 + abs x)) / 2

modelJson :: IndRNN Double
modelJson = unsafePerformIO $ do
  content <- BL.readFile "model.json"
  return $ case eitherDecode content of
    Left err -> error err
    Right x -> x

data IndRNN a = IndRNN {
    weights :: Vector (Vector a)
  , u_layers :: Vector (Vector a)
  , biases :: Vector (Vector a)
  , out_weights :: Vector a
  , out_biases :: Vector a }
  deriving ( Eq, Ord, Show, Functor, Traversable, Foldable, FromJSON, ToJSON, Generic )

data IndRNNState a = IndRNNState {
    activations :: Vector (Vector a)
  , timestep :: Int
  , parentNn :: !(IndRNN a)
  }
  deriving ( Eq, Ord, Show, Functor, Traversable, Foldable )

nlayers :: IndRNN a -> Int
nlayers rnn = V.length (weights rnn) + 2

startV2 :: Num a => IndRNN a -> IndRNNState a
startV2 rnn = IndRNNState
  { activations = V.generate (nlayers rnn - 2) $ \layer_idx ->
                    V.replicate (V.length (u_layers rnn ! layer_idx)) 0
  , parentNn = rnn
  , timestep = 0 }

propagateV2 :: forall a. (Num a, Floating a, Show a) => IndRNNState a -> Vector a -> (IndRNNState a, Vector a)
propagateV2 st inputs =
  let (out, activations) = go 0 inputs
   in ( st { activations = V.fromList activations, timestep = timestep st + 1 }
      , out)
 where
  nn = parentNn st

  go :: Int -> Vector a -> (Vector a, [Vector a]) -- (outputs, activations)
  go layer_idx inputs | layer_idx >= V.length (biases nn) =
    let tgt_layer_sz = V.length (out_biases nn)
        src_layer_sz = V.length inputs
        outs = V.generate tgt_layer_sz $ \target_idx ->
                 let s = sum $ flip fmap [0..src_layer_sz-1] $ \source_idx ->
                           out_weights nn ! (source_idx + target_idx * src_layer_sz) * (inputs ! source_idx)
                     s' = s + (out_biases nn ! target_idx)
                  in fastSigmoid s'

     in (outs, [])

  go layer_idx inputs =
    let src_layer_sz = V.length inputs
        tgt_layer_sz = V.length (biases nn ! layer_idx)

        wgts = weights nn ! layer_idx
        last_acts = activations st ! layer_idx
        ulrs = u_layers nn ! layer_idx
        bis = biases nn ! layer_idx

        new_activations = V.generate tgt_layer_sz $ \target_idx ->
                            let s = sum $ flip fmap [0..src_layer_sz-1] $ \source_idx ->
                                      wgts ! (source_idx + target_idx * src_layer_sz) * (inputs ! source_idx)
                                s' = s + (bis ! target_idx)
                                s'' = s' + (ulrs ! target_idx) * (last_acts ! target_idx)
                             in fastSigmoid s''

        (outputs, rest_activations) = go (layer_idx+1) new_activations

     in (outputs, new_activations:rest_activations)

foldI :: Num a => Int -> (Int -> a) -> a
foldI num_items action = go 0
 where
  go i | i < num_items = action i + go (i+1)
       | otherwise = 0

main :: IO ()
main = do
  --print nn
  let st = startV2 nn
      (_st, out) = propagateV2 st (V.fromList [1.2, 2.4])
      (_st2, out2) = propagateV2 _st (V.fromList [1.3, 2.5])
      (_st3, out3) = propagateV2 _st (V.fromList [1.7, -2.0])

  --print out
  --print out2

  --print $ snd $ propagate st (V.fromList [0.311, 0.422])
  let g = grad (\nn -> let st = startV2 nn
                           (st2, v) = propagateV2 st (V.fromList [1.2, 2.4])
                           (st3, v2) = propagateV2 st2 (V.fromList [1.3, 2.5])
                        in V.sum (snd $ propagateV2 st3 (V.fromList [1.7, -2.0])) +
                           V.sum v + V.sum v2) nn
  print g
  print out2
 where
  nn = modelJson
  st = startV2 nn
