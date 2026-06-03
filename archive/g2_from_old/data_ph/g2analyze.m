function g2analyze 

    % some constants
    tyEmpty8      = hex2dec('FFFF0008');
    tyBool8       = hex2dec('00000008');
    tyInt8        = hex2dec('10000008');
    tyBitSet64    = hex2dec('11000008');
    tyColor8      = hex2dec('12000008');
    tyFloat8      = hex2dec('20000008');
    tyTDateTime   = hex2dec('21000008');
    tyFloat8Array = hex2dec('2001FFFF');
    tyAnsiString  = hex2dec('4001FFFF');
    tyWideString  = hex2dec('4002FFFF');
    tyBinaryBlob  = hex2dec('FFFFFFFF');
    % RecordTypes
    rtPicoHarpT3     = hex2dec('00010303');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $03 (PicoHarp)
    rtPicoHarpT2     = hex2dec('00010203');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $03 (PicoHarp)
    rtHydraHarpT3    = hex2dec('00010304');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $04 (HydraHarp)
    rtHydraHarpT2    = hex2dec('00010204');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $04 (HydraHarp)
    rtHydraHarp2T3   = hex2dec('01010304');% (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $03 (T3), HW: $04 (HydraHarp)
    rtHydraHarp2T2   = hex2dec('01010204');% (SubID = $01 ,RecFmt: $01) (V2), T-Mode: $02 (T2), HW: $04 (HydraHarp)
    rtTimeHarp260NT3 = hex2dec('00010305');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $05 (TimeHarp260N)
    rtTimeHarp260NT2 = hex2dec('00010205');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $05 (TimeHarp260N)
    rtTimeHarp260PT3 = hex2dec('00010306');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $06 (TimeHarp260P)
    rtTimeHarp260PT2 = hex2dec('00010206');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $06 (TimeHarp260P)
    rtMultiHarpNT3   = hex2dec('00010307');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $03 (T3), HW: $07 (MultiHarp150N)
    rtMultiHarpNT2   = hex2dec('00010207');% (SubID = $00 ,RecFmt: $01) (V1), T-Mode: $02 (T2), HW: $07 (MultiHarp150N)

    % Globals for subroutines
    global data;
    global filename;
    global fid;
    global TTResultFormat_TTTRRecType;
    global TTResult_NumberOfRecords; % Number of TTTR Records in the File;
    global MeasDesc_Resolution;      % Resolution for the Dtime (T3 Only)
    global MeasDesc_GlobalResolution;

    TTResultFormat_TTTRRecType = 0;
    TTResult_NumberOfRecords = 0;
    MeasDesc_Resolution = 0;
    MeasDesc_GlobalResolution = 0;

    % start Main program
    [filename, pathname]=uigetfile('*.ptu', 'T-Mode data:');
    fid=fopen([pathname filename]);

    fprintf(1,'\n');
    Magic = fread(fid, 8, '*char');
    if not(strcmp(Magic(Magic~=0)','PQTTTR'))
        error('This is not an PTU file.');
    end;
    Version = fread(fid, 8, '*char');
    fprintf(1,'Tag Version: %s\n', Version);

    % there is no repeat.. until (or do..while) construct in matlab so we use
    % while 1 ... if (expr) break; end; end;
    while 1
        % read Tag Head
        TagIdent = fread(fid, 32, '*char'); % TagHead.Ident
        TagIdent = (TagIdent(TagIdent ~= 0))'; % remove #0 and more more readable
        TagIdx = fread(fid, 1, 'int32');    % TagHead.Idx
        TagTyp = fread(fid, 1, 'uint32');   % TagHead.Typ
                                            % TagHead.Value will be read in the
                                            % right type function
        TagIdent = genvarname(TagIdent);    % remove all illegal characters
        if TagIdx > -1
          EvalName = [TagIdent '(' int2str(TagIdx + 1) ')'];
        else
          EvalName = TagIdent;
        end
        fprintf(1,'\n   %-40s', EvalName);
        % check Typ of Header
        switch TagTyp
            case tyEmpty8
                fread(fid, 1, 'int64');
                fprintf(1,'<Empty>');
            case tyBool8
                TagInt = fread(fid, 1, 'int64');
                if TagInt==0
                    fprintf(1,'FALSE');
                    eval([EvalName '=false;']);
                else
                    fprintf(1,'TRUE');
                    eval([EvalName '=true;']);
                end
            case tyInt8
                TagInt = fread(fid, 1, 'int64');
                fprintf(1,'%d', TagInt);
                eval([EvalName '=TagInt;']);
            case tyBitSet64
                TagInt = fread(fid, 1, 'int64');
                fprintf(1,'%X', TagInt);
                eval([EvalName '=TagInt;']);
            case tyColor8
                TagInt = fread(fid, 1, 'int64');
                fprintf(1,'%X', TagInt);
                eval([EvalName '=TagInt;']);
            case tyFloat8
                TagFloat = fread(fid, 1, 'double');
                fprintf(1, '%e', TagFloat);
                eval([EvalName '=TagFloat;']);
            case tyFloat8Array
                TagInt = fread(fid, 1, 'int64');
                fprintf(1,'<Float array with %d Entries>', TagInt / 8);
                fseek(fid, TagInt, 'cof');
            case tyTDateTime
                TagFloat = fread(fid, 1, 'double');
                fprintf(1, '%s', datestr(datenum(1899,12,30)+TagFloat)); % display as Matlab Date String
                eval([EvalName '=datenum(1899,12,30)+TagFloat;']); % but keep in memory as Matlab Date Number
            case tyAnsiString
                TagInt = fread(fid, 1, 'int64');
                TagString = fread(fid, TagInt, '*char');
                TagString = (TagString(TagString ~= 0))';
                fprintf(1, '%s', TagString);
                if TagIdx > -1
                   EvalName = [TagIdent '{' int2str(TagIdx + 1) '}'];
                end;
                eval([EvalName '=[TagString];']);
            case tyWideString
                % Matlab does not support Widestrings at all, just read and
                % remove the 0's (up to current (2012))
                TagInt = fread(fid, 1, 'int64');
                TagString = fread(fid, TagInt, '*char');
                TagString = (TagString(TagString ~= 0))';
                fprintf(1, '%s', TagString);
                if TagIdx > -1
                   EvalName = [TagIdent '{' int2str(TagIdx + 1) '}'];
                end;
                eval([EvalName '=[TagString];']);
            case tyBinaryBlob
                TagInt = fread(fid, 1, 'int64');
                fprintf(1,'<Binary Blob with %d Bytes>', TagInt);
                fseek(fid, TagInt, 'cof');
            otherwise
                error('Illegal Type identifier found! Broken file?');
        end;
        if strcmp(TagIdent, 'Header_End')
            break
        end
    end
    fprintf(1, '\n----------------------\n');
    outfile = [pathname filename(1:length(filename)-4) '.out'];
    global data;
 
    % Check recordtype
    global isT2;
    switch TTResultFormat_TTTRRecType;
        case rtPicoHarpT2
            isT2 = true;
            fprintf(1,'PicoHarp T2 data\n');
        otherwise
            error('Illegal RecordType!');
    end;
    
    global cnt_ph;
    global cnt_ov;
    global cnt_ma;
    cnt_ph = 0;
    cnt_ov = 0;
    cnt_ma = 0;
    % choose right decode function
    switch TTResultFormat_TTTRRecType;
        case rtPicoHarpT2
            isT2 = true;
            ReadPT2;
        otherwise
            error('Illegal RecordType!');
    end;
    fclose(fid);
    fprintf(1,'\n');

end


%% This function is to write records into data file

function ReadPT2
    global fid;
    global data;
    global TTResult_NumberOfRecords; % Number of TTTR Records in the File;
    %TTResult_NumberOfRecords=round(TTResult_NumberOfRecords*0.5);
    ofltime = 0;
    data =zeros(TTResult_NumberOfRecords,2);
    WRAPAROUND=210698240;
    j=1;
    for i=1:(TTResult_NumberOfRecords)
        T2Record = fread(fid, 1, 'ubit32');
        T2time = bitand(T2Record,268435455);             %the lowest 28 bits
        chan = bitand(bitshift(T2Record,-28),15);      %the next 4 bits
        timetag = T2time + ofltime;
        if (chan >= 0) 
            if(chan <= 4)
            data(j,1)=chan;
            data(j,2)=timetag*4;
            j=j+1;
            elseif chan == 15
                markers = bitand(T2Record,15);  % where the lowest 4 bits are marker bits
                if markers==0                   % then this is an overflow record
                    ofltime = ofltime + WRAPAROUND; % and we unwrap the time tag overflow
                    GotOverflow(1);
                else                            % otherwise it is a true marker
                    GotMarker(timetag, markers);
                end;
            end;
        end;
        
    end;
end

%% Got Overflow
%  Count: Some TCSPC provide Overflow compression = if no Photons between overflow you get one record for multiple Overflows
function GotOverflow(Count)
  global cnt_ov;
  cnt_ov = cnt_ov + Count;

end

%% Got Marker
%    TimeTag: Raw TimeTag from Record * Globalresolution = Real Time arrival of Photon
%    Markers: Bitfield of arrived Markers, different markers can arrive at same time (same record)
function GotMarker(TimeTag, Markers)
  global cnt_ma;
  global MeasDesc_GlobalResolution;
  cnt_ma = cnt_ma + 1;
end


