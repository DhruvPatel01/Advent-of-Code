module Day07 where

import qualified Data.Set as S
import qualified Data.List as L
import qualified Data.Map.Strict as M

input = lines <$> readFile "./inputs/day07.txt"

part1_stub (c, tachyons) splitters = (c+count, new_tachyons)
    where split_at = S.fromDistinctAscList [i | (i, s) <- zip [0..] splitters, s == '^']
          count = length . filter id . map (`S.member` tachyons) . S.elems $ split_at
          new_temp = S.fromList . concatMap (\x -> [x-1, x+1]) $ split_at
          new_tachyons = S.difference (S.union new_temp tachyons) split_at

part1 start splitterss = fst final
    where 
        start_tachyons = S.fromList [i | (i, s) <- zip [0..] start, s == 'S'] 
        final = L.foldl' part1_stub (0, start_tachyons) splitterss


part2_stub counts splitters = new
    where  
        split_at =  [i | (i, s) <- zip [0..] splitters, s == '^']
        to_add_to_left = M.fromList . filter ((> 0) . snd) $ map (\s -> (s-1, M.findWithDefault 0 s counts)) split_at
        to_add_to_right = M.fromList . filter ((> 0) . snd) $ map (\s -> (s+1, M.findWithDefault 0 s counts)) split_at
        new_counts = M.unionsWith (+) [counts, to_add_to_left, to_add_to_right]
        new = L.foldl' (flip M.delete) new_counts split_at

part2 start splitterss = sum . M.elems $ final
    where
        startWith = M.fromList [(i, 1) | (i, s) <- zip [0..] start, s == 'S'] 
        final = L.foldl' part2_stub startWith splitterss

main = do
    (start:splitterss) <- input
    print $ part1 start splitterss
    print $ part2 start splitterss
