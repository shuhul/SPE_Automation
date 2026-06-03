function g2analyze_headless(ptu_path, max_records)
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
    rtPicoHarpT2  = hex2dec('00010203');

    global data;
    global filename;
    global fid;
    global TTResultFormat_TTTRRecType;
    global TTResult_NumberOfRecords;
    global MeasDesc_Resolution;
    global MeasDesc_GlobalResolution;
    global isT2;
    global cnt_ph;
    global cnt_ov;
    global cnt_ma;

    TTResultFormat_TTTRRecType = 0;
    TTResult_NumberOfRecords = 0;
    MeasDesc_Resolution = 0;
    MeasDesc_GlobalResolution = 0;

    [pathname, name, ext] = fileparts(ptu_path);
    filename = [name ext];
    if isempty(pathname); pathname = '.'; end
    fid = fopen(ptu_path);

    Magic = fread(fid, 8, '*char');
    if not(strcmp(Magic(Magic~=0)','PQTTTR'))
        error('Not a PTU file.');
    end
    Version = fread(fid, 8, '*char'); %#ok<NASGU>

    while 1
        TagIdent = fread(fid, 32, '*char');
        TagIdent = (TagIdent(TagIdent ~= 0))';
        TagIdx = fread(fid, 1, 'int32');
        TagTyp = fread(fid, 1, 'uint32');
        TagIdent = genvarname(TagIdent);
        if TagIdx > -1
            EvalName = [TagIdent '(' int2str(TagIdx + 1) ')'];
        else
            EvalName = TagIdent;
        end
        switch TagTyp
            case tyEmpty8
                fread(fid, 1, 'int64');
            case tyBool8
                TagInt = fread(fid, 1, 'int64');
                if TagInt==0
                    eval([EvalName '=false;']);
                else
                    eval([EvalName '=true;']);
                end
            case tyInt8
                TagInt = fread(fid, 1, 'int64');
                eval([EvalName '=TagInt;']);
            case tyBitSet64
                TagInt = fread(fid, 1, 'int64');
                eval([EvalName '=TagInt;']);
            case tyColor8
                TagInt = fread(fid, 1, 'int64');
                eval([EvalName '=TagInt;']);
            case tyFloat8
                TagFloat = fread(fid, 1, 'double');
                eval([EvalName '=TagFloat;']);
            case tyFloat8Array
                TagInt = fread(fid, 1, 'int64');
                fseek(fid, TagInt, 'cof');
            case tyTDateTime
                TagFloat = fread(fid, 1, 'double');
                eval([EvalName '=datenum(1899,12,30)+TagFloat;']);
            case tyAnsiString
                TagInt = fread(fid, 1, 'int64');
                TagString = fread(fid, TagInt, '*char');
                TagString = (TagString(TagString ~= 0))';
                if TagIdx > -1
                    EvalName = [TagIdent '{' int2str(TagIdx + 1) '}'];
                end
                eval([EvalName '=[TagString];']);
            case tyWideString
                TagInt = fread(fid, 1, 'int64');
                TagString = fread(fid, TagInt, '*char');
                TagString = (TagString(TagString ~= 0))';
                if TagIdx > -1
                    EvalName = [TagIdent '{' int2str(TagIdx + 1) '}'];
                end
                eval([EvalName '=[TagString];']);
            case tyBinaryBlob
                TagInt = fread(fid, 1, 'int64');
                fseek(fid, TagInt, 'cof');
            otherwise
                error('Illegal Type identifier!');
        end
        if strcmp(TagIdent, 'Header_End')
            break
        end
    end

    if TTResultFormat_TTTRRecType ~= rtPicoHarpT2
        error('Only PicoHarp T2 supported');
    end
    isT2 = true;

    if nargin >= 2 && ~isempty(max_records)
        TTResult_NumberOfRecords = min(TTResult_NumberOfRecords, max_records);
    end

    cnt_ph = 0; cnt_ov = 0; cnt_ma = 0;
    ReadPT2();
    fclose(fid);
end

function ReadPT2()
    global fid;
    global data;
    global TTResult_NumberOfRecords;
    ofltime = 0;
    data = zeros(TTResult_NumberOfRecords, 2);
    WRAPAROUND = 210698240;
    j = 1;
    for i = 1:TTResult_NumberOfRecords
        T2Record = fread(fid, 1, 'ubit32');
        T2time = bitand(T2Record, 268435455);
        chan = bitand(bitshift(T2Record, -28), 15);
        timetag = T2time + ofltime;
        if chan >= 0
            if chan <= 4
                data(j,1) = chan;
                data(j,2) = timetag * 4;
                j = j + 1;
            elseif chan == 15
                markers = bitand(T2Record, 15);
                if markers == 0
                    ofltime = ofltime + WRAPAROUND;
                end
            end
        end
    end
end
